"""Video Text on Surface: text, a price or a legal super that looks printed on a moving surface.

Pipeline (CPU: OpenCV + numpy + Pillow; no model weights):
1. Reference frame: the frame where most of the surface is visible and sharp (or a given one).
2. Track: one homography per frame maps the reference frame onto the surface. All alignment runs on a band-pass of
   log intensity, so light changes (flares, moving soft shadows) don't count as motion. Corner points tracked from
   the previous frame (KLT, forward-backward checked) predict an affine step; ECC refines it against an anchor frame,
   first as an affine correction on the pre-warped frame (well-conditioned on thin strips), then as a full
   homography when the visible patch is two-dimensional. A robust pass drops reflections that slide over the
   surface. The anchor is the reference frame (no drift); while the look drifts, a recent well-aligned frame stands
   in, and the reference is taken back as soon as it aligns again. Texture that runs one way (grain, stripes) keeps
   the corner prediction along that way (aperture problem); a surface with too little detail of its own is tracked
   with its surroundings. When the surface leaves the frame the motion is extrapolated; SIFT finds it again. A
   5-frame fit removes sub-pixel jitter only (never real motion).
3. Layout: the surface's shape picks the warp. Straight, parallel sides and a curved base = Round (bottle, can,
   jar): the text wraps around a cylinder (axis, radius and the ellipse of the base from the matte). A four-sided
   outline = Flat in perspective (box face, board, sign, table): the text lies on that rectangle's plane. Any other
   outline: the text faces the reference frame's view. Position (Center, Top, ..., Emptiest Area) and size are
   measured on the surface.
4. Render: text (Pillow/FreeType) as a supersampled texture with mip levels, inverse-mapped per output pixel
   (3x3 supersampling): sub-pixel position, clean antialiasing at any scale and angle. Motion blur from the
   tracked motion (180-degree shutter, sub-frame accumulation).
5. Look: the ink takes the surface's own light: shading = local surface luminance (existing print removed by a grey
   closing) relative to its typical level over the clip, the light's colour shift, and the surface's fine texture.
   The colour given is the colour under the clip's typical light; on a light surface the ink is never brighter than
   the surface's own white; empty colour = contrasting ink. The matte decides where the surface is visible, so
   anything in front of it hides the text; an optional check also finds occluders the matte misses (by look against
   the reference frame and by motion relative to the surface). Pixels outside the text come back unchanged, bit
   for bit.

The node (AutoMatteTextOnSurface) is at the bottom; everything above it is plain numpy and also runs outside ComfyUI.
Licences: OpenCV (Apache 2.0, SIFT's patent has expired), numpy (BSD), Pillow (MIT-CMU); fonts OFL (fonts/).
"""

import json
import math
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ANALYSIS_LONG = 1024          # tracking resolution (long side, px)
SHUTTER = 0.5                 # motion-blur shutter as a fraction of the frame interval (180 degrees)
LOW_DETAIL = 1.2              # band-pass contrast below which a surface is tracked with its surroundings
ONE_WAY = 0.12                # texture strength ratio (weak/strong direction) below which the texture runs one way
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

# Bundled fonts (all SIL Open Font License 1.1). name -> (regular file, bold file)
# Bundled fonts (all SIL Open Font License 1.1, files as named in github.com/google/fonts/tree/main/ofl/<family>;
# list with URLs and sizes in fonts_to_fetch.md). name -> {"regular"|"bold": (file, weight on a variable font or None)}
_V = lambda f: {"regular": (f, 400), "bold": (f, 700)}
FONTS = {
    "Inter": _V("Inter[opsz,wght].ttf"),
    "Montserrat": _V("Montserrat[wght].ttf"),
    "Lato": {"regular": ("Lato-Regular.ttf", None), "bold": ("Lato-Bold.ttf", None)},
    "Poppins": {"regular": ("Poppins-Regular.ttf", None), "bold": ("Poppins-Bold.ttf", None)},
    "Oswald": _V("Oswald[wght].ttf"),
    "Bebas Neue": {"regular": ("BebasNeue-Regular.ttf", None), "bold": ("BebasNeue-Regular.ttf", None)},
    "Playfair Display": _V("PlayfairDisplay[wght].ttf"),
    "Libre Baskerville": _V("LibreBaskerville[wght].ttf"),
    "Noto Sans": _V("NotoSans[wdth,wght].ttf"),
    "Noto Serif": _V("NotoSerif[wdth,wght].ttf"),
}
# Text in these scripts switches to the matching Noto font when it is bundled (Latin fonts don't have the glyphs).
SCRIPT_FONTS = [  # (script, unicode ranges, variable font file, needs shaping)
    ("Japanese", ((0x3040, 0x30FF), (0x31F0, 0x31FF)), "NotoSansJP[wght].ttf", False),
    ("Korean", ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)), "NotoSansKR[wght].ttf", False),
    ("Chinese", ((0x4E00, 0x9FFF), (0x3400, 0x4DBF)), "NotoSansSC[wght].ttf", False),
    ("Arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
     "NotoSansArabic[wdth,wght].ttf", True),
    ("Hebrew", ((0x0590, 0x05FF), (0xFB1D, 0xFB4F)), "NotoSansHebrew[wdth,wght].ttf", True),
    ("Thai", ((0x0E00, 0x0E7F),), "NotoSansThai[wdth,wght].ttf", True),
    ("Devanagari", ((0x0900, 0x097F),), "NotoSansDevanagari[wdth,wght].ttf", True),
]
PLACEMENTS = ["Center", "Top", "Bottom", "Left", "Right", "Top Left", "Top Right", "Bottom Left", "Bottom Right",
              "Emptiest Area"]
SIZES = {"Auto": None, "Small": 0.45, "Medium": 0.65, "Large": 0.85}


# --------------------------------------------------------------------------------------------- small helpers
def srgb_to_lin(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def _lum(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _norm_h(H):
    return H / H[2, 2]


def _apply_h(H, pts):
    """pts (k,2) -> (k,2)."""
    p = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ H.T
    return p[:, :2] / p[:, 2:3]


def _scale_h(H, s):
    """Homography between images scaled by s (both sides): S H S^-1."""
    S = np.diag([s, s, 1.0])
    return S @ H @ np.linalg.inv(S)


def _h_scale(H, p):
    """Local scale (sqrt of the Jacobian determinant) of homography H at point p."""
    q = _apply_h(H, np.array([p, [p[0] + 1, p[1]], [p[0], p[1] + 1]], float))
    return math.sqrt(abs((q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) - (q[2, 0] - q[0, 0]) * (q[1, 1] - q[0, 1])))


def _ana_to_full(H, sx, sy):
    """Homography in analysis px -> full-res px (pixel-centre convention of an INTER_AREA resize)."""
    A = np.array([[sx, 0, 0.5 * (sx - 1)], [0, sy, 0.5 * (sy - 1)], [0, 0, 1.0]])
    return _norm_h(np.linalg.inv(A) @ H @ A)


def _h_ok(H, w, h):
    """Rejects degenerate / folded homographies."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    try:
        q = _apply_h(H, c)
    except Exception:
        return False
    if not np.all(np.isfinite(q)):
        return False
    z = np.concatenate([c, np.ones((4, 1))], 1) @ H.T
    if np.any(z[:, 2] <= 0):
        return False
    a = cv2.contourArea(q.astype(np.float32))
    return 0.02 * w * h < a < 50 * w * h and cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2))


def _odd(k):
    k = max(1, int(round(k)))
    return k if k % 2 else k + 1


def _blur_w(x, w, sigma):
    """Gaussian blur of x weighted by w (normalised): ignores pixels with weight 0."""
    ks = _odd(6 * sigma + 1)
    num = cv2.GaussianBlur(x * (w[..., None] if x.ndim == 3 else w), (ks, ks), sigma)
    den = cv2.GaussianBlur(w, (ks, ks), sigma)
    den = np.maximum(den, 1e-4)
    return num / (den[..., None] if x.ndim == 3 else den)


def parse_colour(text):
    """'#2E7D32', '#abc', 'rgb(46,125,50)' or a CSS name -> ((r, g, b), '#RRGGBB'). The pack's layers.parse_colour
    (CSS + design names) replaces this when the module sits in ComfyUI-AutoMatte."""
    try:
        from .layers import parse_colour as pc  # type: ignore
        return pc(text)
    except Exception:
        pass
    raw = (text or "").strip()
    t = raw.lower()
    m = re.fullmatch(r"rgb\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)", t)
    if m:
        rgb = tuple(min(255, int(v)) for v in m.groups())
    else:
        h = t.lstrip("#")
        names = {"white": "ffffff", "black": "000000", "red": "ff0000", "gold": "ffd700", "navy": "000080",
                 "silver": "c0c0c0", "cream": "fffdd0", "charcoal": "36454f", "brown": "a52a2a"}
        if re.fullmatch(r"[0-9a-f]{6}", h):
            pass
        elif re.fullmatch(r"[0-9a-f]{3}", h):
            h = "".join(c * 2 for c in h)
        elif re.sub(r"[\s\-_]", "", t) in names:
            h = names[re.sub(r"[\s\-_]", "", t)]
        else:
            raise ValueError(f"Colour '{raw}' isn't a hex code (#2E7D32), rgb(46, 125, 50) or a known colour name.")
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return rgb, "#%02X%02X%02X" % rgb


# --------------------------------------------------------------------------------------------------- tracking
def _find_ecc(tmpl, img, W, crit, tmask, gauss=3, motion=cv2.MOTION_HOMOGRAPHY):
    """ECC with a template mask on OpenCV 4.x (findTransformECC's mask = template) and 5.x (the template mask moved
    to findTransformECCWithMask; findTransformECC's mask now belongs to the input image)."""
    W = np.ascontiguousarray(W, np.float32)
    if hasattr(cv2, "findTransformECCWithMask"):
        ones = np.full(img.shape[:2], 255, np.uint8)
        return cv2.findTransformECCWithMask(tmpl, img, tmask, ones, W, motion, crit, gauss)
    return cv2.findTransformECC(tmpl, img, W, motion, crit, tmask, gauss)


HP_SIGMAS = (0.8, 3.0)


def _highpass(g):
    """Alignment image robust to light changes (flares, moving soft shadows): a band-pass of log intensity. In log
    space shading is an offset, which the band-pass removes; soft shadow edges are wider than the band and drop out,
    while print, grain and veins (the surface's own detail) stay."""
    lg = np.log(g.astype(np.float32) + 10.0) * 40.0
    return cv2.GaussianBlur(lg, (0, 0), HP_SIGMAS[0]) - cv2.GaussianBlur(lg, (0, 0), HP_SIGMAS[1])


class SurfaceTracker:
    """Per-frame homography ref -> frame (analysis px) of a surface.

    grays: list of (h, w) uint8 frames at analysis size. roi_ref: (h, w) bool, the part of the surface in the
    reference frame that is tracked. vis: optional per-frame (h, w) float masks of the visible surface (the matte).
    occ: optional per-frame occluder masks.

    Each frame: KLT points from the previous frame predict the motion (RANSAC homography); a dense ECC alignment on
    band-passed images against an anchor frame refines it. The anchor is the reference frame; when the surface's
    look drifts away from it (light change, big zoom) a recent well-aligned frame becomes the anchor, and the
    reference is taken back as soon as it aligns well again (no drift)."""

    KEY_CC = 0.88     # below this against the anchor, a better-matching recent frame may become the anchor
    BACK_CC = 0.85    # the reference frame is used again when it aligns at least this well

    def __init__(self, grays, ref, roi_ref, vis=None, occ=None, log=None):
        self.g = grays
        self.n = len(grays)
        self.h, self.w = grays[0].shape
        self.ref = ref
        self.roi = roi_ref.astype(bool)
        self.vis = vis
        self.occ = occ
        self.log = log or (lambda *a: None)
        self.roi_u8 = self.roi.astype(np.uint8) * 255
        self.roi_dil = cv2.dilate(self.roi_u8, np.ones((7, 7), np.uint8))
        self.roi_area = int(self.roi.sum())
        e = max(5, _odd(0.02 * math.sqrt(max(1, self.roi_area))))
        er = cv2.erode(self.roi_u8, np.ones((e, e), np.uint8))
        self.core_u8 = er if er.sum() > 200 * 255 else self.roi_u8
        self._hp = {}
        self._k8 = {}
        self.last_ratio = 1.0
        hr = _highpass(grays[ref])[self.roi]
        self.k_gain = float(np.clip(28.0 / (hr.std() + 1e-3), 1.0, 12.0))  # faint texture (dark wood) gets contrast
        try:
            self.sift = cv2.SIFT_create(nfeatures=1500)
            self.kp_ref, self.des_ref = self.sift.detectAndCompute(grays[ref], self.roi_u8)
        except Exception:
            self.sift, self.kp_ref, self.des_ref = None, [], None

    def hp(self, t):
        if t not in self._hp:
            if len(self._hp) > 12:
                self._hp.pop(next(iter(self._hp)))
            self._hp[t] = _highpass(self.g[t])
        return self._hp[t]

    def k8(self, t):
        """8-bit contrast-normalised band-pass frame for point tracking (light changes removed)."""
        if t not in self._k8:
            if len(self._k8) > 6:
                self._k8.pop(next(iter(self._k8)))
            self._k8[t] = np.clip(self.hp(t) * self.k_gain + 128, 0, 255).astype(np.uint8)
        return self._k8[t]

    # -- masks ------------------------------------------------------------------------------------------------
    def visible(self, t, H):
        """Visible surface in frame t (bool, frame coords) given the estimate H: the tracked part of the surface
        (reference ROI warped by H, slightly grown), inside the matte when one is given, minus occluders."""
        # with a matte the matte confines the surface; without one, stay inside the drawn surface
        v = cv2.warpPerspective(self.roi_dil if self.vis is not None else self.core_u8, H, (self.w, self.h),
                                flags=cv2.INTER_NEAREST) > 0
        if self.vis is not None:
            v &= self.vis[t] > 0.5
        if self.occ is not None:
            v &= self.occ[t] < 0.5
        v &= self.g[t] < 250  # clipped highlights carry no texture
        return v

    # -- motion estimates -------------------------------------------------------------------------------------
    def _klt(self, a, b, Ha):
        m = self.visible(a, Ha).astype(np.uint8)
        m = cv2.erode(m, np.ones((5, 5), np.uint8)) * 255
        if m.sum() < 255 * 200:
            return None, 0
        ga, gb = self.k8(a), self.k8(b)
        p0 = cv2.goodFeaturesToTrack(ga, maxCorners=900, qualityLevel=0.002, minDistance=5, mask=m, blockSize=7)
        if p0 is None or len(p0) < 12:
            return None, 0
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, p0, None, winSize=(21, 21), maxLevel=4, criteria=crit)
        if p1 is None:
            return None, 0
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gb, ga, p1, None, winSize=(21, 21), maxLevel=4, criteria=crit)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.8)
        if good.sum() < 10:
            return None, int(good.sum())
        # one frame of motion: an affine step is enough to predict, and stays well-conditioned when the points sit
        # on a thin strip (the dense ECC that follows refines to a full homography)
        A, inl = cv2.estimateAffine2D(p0[good].reshape(-1, 2), p1[good].reshape(-1, 2), method=cv2.RANSAC,
                                      ransacReprojThreshold=1.5, maxIters=3000, confidence=0.995)
        if A is None or inl is None or inl.sum() < 8:
            return None, 0
        return np.vstack([A, [0, 0, 1.0]]), int(inl.sum())

    def _sift(self, b):
        if self.sift is None or self.des_ref is None or len(self.kp_ref) < 8:
            return None
        m = (self.g[b] < 250).astype(np.uint8) * 255
        if self.vis is not None:
            m = cv2.dilate(((self.vis[b] > 0.5) * 255).astype(np.uint8), np.ones((9, 9), np.uint8))
        kp, des = self.sift.detectAndCompute(self.g[b], m)
        if des is None or len(kp) < 8:
            return None
        mt = cv2.BFMatcher(cv2.NORM_L2).knnMatch(self.des_ref, des, k=2)
        good = [x[0] for x in mt if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
        if len(good) < 12:
            return None
        src = np.float32([self.kp_ref[x.queryIdx].pt for x in good])
        dst = np.float32([kp[x.trainIdx].pt for x in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=5000, confidence=0.999)
        if H is None or inl.sum() < 12 or not _h_ok(H, self.w, self.h):
            return None
        return _norm_h(H)

    def _ecc_to(self, a, Ha, b, H0):
        """Align frame b onto anchor frame a (ref -> a is Ha), starting from H0 (ref -> b).
        Returns (H ref -> b, cc, mask px) or (None, 0, px)."""
        W0 = _norm_h(H0 @ np.linalg.inv(Ha))                      # a -> b
        if not _h_ok(W0, self.w, self.h):
            return None, 0.0, 0
        core_a = cv2.warpPerspective(self.core_u8, Ha, (self.w, self.h), flags=cv2.INTER_NEAREST) > 0
        vb = self.visible(b, H0).astype(np.uint8) * 255
        back = cv2.warpPerspective(vb, W0, (self.w, self.h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
        m = core_a & self.visible(a, Ha) & back
        m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8))
        npx = int(m.sum())
        if npx < max(300, 0.03 * self.roi_area):
            return None, 0.0, npx
        ys, xs = np.nonzero(m)
        x0, y0 = max(0, xs.min() - 8), max(0, ys.min() - 8)
        x1, y1 = min(self.w, xs.max() + 9), min(self.h, ys.max() + 9)
        T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1.0]])
        tmpl = np.ascontiguousarray(self.hp(a)[y0:y1, x0:x1])
        img = self.hp(b)
        mc = np.ascontiguousarray(m[y0:y1, x0:x1])
        sc = _h_scale(W0, ((x0 + x1) / 2, (y0 + y1) / 2))
        if sc < 0.85:     # b smaller than the anchor: soften the template to b's detail
            tmpl = cv2.GaussianBlur(tmpl, (0, 0), 0.5 * math.sqrt(1 / (sc * sc) - 1))
        elif sc > 1.18:   # b bigger: soften b so it isn't aliased when sampled at the anchor's pixels
            img = cv2.GaussianBlur(img, (0, 0), 0.5 * math.sqrt(sc * sc - 1))
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
        Wc0 = W0 @ T                                              # crop coords -> b
        th_, tw_ = tmpl.shape
        # 1) warp b onto the template with the prediction, then a small affine correction (well-conditioned even
        #    on a thin strip or a surface with one-directional texture such as wood grain)
        try:
            Iw = cv2.warpPerspective(img, Wc0, (tw_, th_), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            cc, A = _find_ecc(tmpl, Iw, np.eye(3, dtype=np.float32)[:2], crit, mc, motion=cv2.MOTION_AFFINE)
            Wc = Wc0 @ np.vstack([A.astype(np.float64), [0, 0, 1.0]])
        except cv2.error:
            return None, 0.0, npx
        # 2) full homography refinement, only when the visible patch is two-dimensional enough
        ys_, xs_ = np.nonzero(mc)
        ext = (ys_.max() - ys_.min() + 1) / max(1.0, xs_.max() - xs_.min() + 1)
        if 0.2 < ext < 5.0:
            try:
                cc2, W2 = _find_ecc(tmpl, img, Wc.astype(np.float32), crit, mc)
                W2 = W2.astype(np.float64)
                if cc2 >= cc - 0.002 and _h_ok(_norm_h(W2 @ np.linalg.inv(T) @ Ha), self.w, self.h):
                    cc, Wc = cc2, W2
            except cv2.error:
                pass
        # 3) robust pass: drop pixels that don't follow the surface (moving reflections, flares, occluders)
        try:
            warped = cv2.warpPerspective(img, Wc, (tw_, th_), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            mk = mc > 0
            ta, tb = tmpl[mk], warped[mk]
            ta = (ta - ta.mean()) / (ta.std() + 1e-6)
            tb = (tb - tb.mean()) / (tb.std() + 1e-6)
            r = np.abs(ta - tb)
            keep = np.zeros_like(mc)
            keep[mk] = (r <= np.percentile(r, 85)).astype(np.uint8)
            if int(keep.sum()) > 300:
                Iw = cv2.warpPerspective(img, Wc, (tw_, th_), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                cc3, A = _find_ecc(tmpl, Iw, np.eye(3, dtype=np.float32)[:2], crit, keep, motion=cv2.MOTION_AFFINE)
                Wc = Wc @ np.vstack([A.astype(np.float64), [0, 0, 1.0]])
                cc = cc3
        except cv2.error:
            pass
        # aperture problem: texture that runs one way (wood grain, stripes, a thin strip) doesn't pin the motion
        # along that way -> there the prediction (tracked corners) is kept and ECC only corrects across it
        gx = cv2.Sobel(tmpl, cv2.CV_32F, 1, 0, ksize=3)[mc > 0]
        gy = cv2.Sobel(tmpl, cv2.CV_32F, 0, 1, ksize=3)[mc > 0]
        J = np.array([[np.dot(gx, gx), np.dot(gx, gy)], [np.dot(gx, gy), np.dot(gy, gy)]], np.float64)
        ev, evec = np.linalg.eigh(J)
        ratio = float(ev[0] / max(ev[1], 1e-9))
        self.last_ratio = ratio
        if ratio < ONE_WAY:
            cxy = np.array([[xs.mean() - x0, ys.mean() - y0]])
            d = (_apply_h(Wc, cxy) - _apply_h(Wc0, cxy))[0]
            wdir = evec[:, 0]                                     # weak direction (smallest eigenvalue)
            k_ = 1.0 - ratio / ONE_WAY
            sh = -np.dot(d, wdir) * wdir * k_
            Wc = np.array([[1, 0, sh[0]], [0, 1, sh[1]], [0, 0, 1.0]]) @ Wc
        W = _norm_h(Wc.astype(np.float64) @ np.linalg.inv(T))
        H = _norm_h(W @ Ha)
        if not _h_ok(H, self.w, self.h):
            return None, 0.0, npx
        return H, float(cc), npx

    def _roi_corners(self):
        ys, xs = np.nonzero(self.roi)
        return np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], float)

    # -- main loop --------------------------------------------------------------------------------------------
    def run(self):
        n, ref = self.n, self.ref
        H = [None] * n
        info = [None] * n
        H[ref] = np.eye(3)
        info[ref] = {"status": "reference", "cc": 1.0, "klt": -1, "anchor": ref}
        corners = self._roi_corners()
        diag = float(np.linalg.norm(corners[2] - corners[0]))
        ys, xs = np.nonzero(self.core_u8)
        pick = np.linspace(0, len(xs) - 1, min(400, len(xs))).astype(int)
        samples = np.stack([xs[pick], ys[pick]], 1).astype(float)

        def agree(Ha_, Hb_, frac):
            """Two estimates agree on the part of the surface that is in frame (not at off-screen corners)."""
            q = _apply_h(Hb_, samples)
            inside = (q[:, 0] >= 0) & (q[:, 0] < self.w) & (q[:, 1] >= 0) & (q[:, 1] < self.h)
            pts = samples[inside] if inside.sum() >= 10 else corners
            sb = _h_scale(Hb_, pts.mean(0))
            ext = float(np.linalg.norm(pts.max(0) - pts.min(0))) if inside.sum() >= 10 else diag
            return np.abs(_apply_h(Ha_, pts) - _apply_h(Hb_, pts)).max() <= max(6.0, frac * max(ext, 0.25 * diag) * sb)

        for direction in (1, -1):
            prev2, a, anchor, since_ref = None, ref, ref, 0
            for b in range(ref + direction, n if direction > 0 else -1, direction):
                Ha = H[a]
                in_view = cv2.warpPerspective(self.core_u8, Ha, (self.w, self.h), flags=cv2.INTER_NEAREST).sum() \
                    / 255.0 / max(1, int(self.core_u8.sum() / 255))
                G, nin = self._klt(a, b, Ha) if in_view >= 0.08 else (None, 0)
                if G is not None:
                    Hp, src = _norm_h(G @ Ha), "klt"
                elif prev2 is not None:
                    Hp, src = _norm_h(Ha @ np.linalg.inv(prev2) @ Ha), "velocity"
                else:
                    Hp, src = Ha.copy(), "hold"
                if not _h_ok(Hp, self.w, self.h):
                    Hp, src = Ha.copy(), "hold"
                frac = 0.08 if src == "klt" else 0.25
                He, cc, npx = self._ecc_to(anchor, H[anchor], b, Hp)
                if He is not None and not agree(He, Hp, frac):
                    He, cc = None, 0.0
                # back to the reference whenever it aligns well again (drift reset)
                since_ref += 1
                if anchor != ref and (since_ref % 4 == 0 or He is None or cc < self.KEY_CC):
                    Hr, ccr, _ = self._ecc_to(ref, H[ref], b, He if He is not None else Hp)
                    if Hr is not None and ccr >= self.BACK_CC and agree(Hr, He if He is not None else Hp, frac):
                        He, cc, anchor, since_ref = Hr, ccr, ref, 0
                # the look drifted: a recent frame that matches better becomes the anchor
                if (He is None or cc < self.KEY_CC) and a != anchor and info[a]["status"] in ("tracked", "reacquired"):
                    H2, cc2, _ = self._ecc_to(a, H[a], b, He if He is not None else Hp)
                    if H2 is not None and agree(H2, Hp, frac) and cc2 > (cc + 0.04 if He is not None else 0.5):
                        He, cc, anchor = H2, cc2, a
                status = "tracked"
                if He is None:
                    Hs = self._sift(b)
                    if Hs is not None:
                        H3, cc3, npx = self._ecc_to(ref, H[ref], b, Hs)
                        if H3 is not None and cc3 >= 0.5:
                            He, cc, status, anchor = H3, cc3, "reacquired", ref
                if He is None:
                    He = Hp
                    status = "predicted" if src != "hold" else "held"
                    if int(self.visible(b, Hp).sum()) < 0.02 * self.roi_area:
                        status = "out_of_view"
                H[b] = He
                info[b] = {"status": status, "cc": round(float(cc), 4), "klt": int(nin), "src": src, "px": npx,
                           "anchor": int(anchor), "texture_2d": round(self.last_ratio, 3)}
                prev2, a = Ha, b
            self.log(f"tracked {'forward' if direction > 0 else 'backward'} from frame {ref}")
        return np.stack(H), info


def smooth_track(H, info, pts, noise_px=0.35):
    """Removes sub-pixel jitter: the tracked points' paths get a 5-frame quadratic (Savitzky-Golay) fit, applied
    only where it moves a point by less than noise_px (larger differences are real motion and stay)."""
    n = len(H)
    if n < 5:
        return H
    P = np.stack([_apply_h(H[t], pts) for t in range(n)])           # (n, k, 2)
    ok = np.array([i["status"] in ("tracked", "reference", "reacquired") for i in info])
    c = np.array([-3, 12, 17, 12, -3], float) / 35
    S = P.copy()
    for t in range(2, n - 2):
        if ok[t - 2:t + 3].all():
            S[t] = np.tensordot(c, P[t - 2:t + 3], axes=(0, 0))
    d = np.linalg.norm(S - P, axis=2).max(1)
    out = H.copy()
    src = pts[:4].astype(np.float32)
    for t in range(n):
        if 0 < d[t] < noise_px:
            out[t] = cv2.getPerspectiveTransform(src, S[t, :4].astype(np.float32)).astype(np.float64)
    return out


# ------------------------------------------------------------------------------------------------------ shape
def fit_round(matte_ref):
    """Cylinder test on a reference-frame matte (full res, bool). Returns a dict with kind and parameters."""
    m = matte_ref.astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if num < 2:
        return {"kind": "none", "reason": "empty matte"}
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = lab == k
    ys = np.nonzero(m.any(1))[0]
    rows = []
    for y in range(ys.min(), ys.max() + 1):
        xs = np.nonzero(m[y])[0]
        if len(xs) < 3:
            continue
        xl, xr = xs.min(), xs.max()
        rows.append((y, xl, xr, len(xs) / (xr - xl + 1)))
    rows = np.array(rows, float)
    width = rows[:, 2] - rows[:, 1]
    wmax = np.percentile(width, 97)
    body = (width >= 0.88 * wmax) & (rows[:, 3] >= 0.93)
    # longest contiguous run of body rows
    best, cur, start, bs = 0, 0, 0, 0
    for i, ok in enumerate(body):
        if ok:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, bs = cur, start
        else:
            cur = 0
    if best < 10:
        return {"kind": "flat", "reason": "no straight-sided body"}
    sl = slice(bs, bs + best)
    yb, xl, xr = rows[sl, 0], rows[sl, 1], rows[sl, 2]
    body_h = yb[-1] - yb[0]
    if body_h < 0.6 * wmax:
        return {"kind": "flat", "reason": "body too short for a cylinder"}
    trim = slice(int(0.1 * best), max(int(0.1 * best) + 5, int(0.9 * best)))
    pl, pr = np.polyfit(yb[trim], xl[trim], 1), np.polyfit(yb[trim], xr[trim], 1)
    res = max(np.sqrt(np.mean((np.polyval(pl, yb[trim]) - xl[trim]) ** 2)),
              np.sqrt(np.mean((np.polyval(pr, yb[trim]) - xr[trim]) ** 2)))
    straight = res < 0.015 * wmax
    parallel = abs(pl[0] - pr[0]) < 0.06
    # base arc: bottom boundary of the body columns
    y_end = int(yb[-1])
    xc_b = 0.5 * (np.polyval(pl, y_end) + np.polyval(pr, y_end))
    r_b = 0.5 * (np.polyval(pr, y_end) - np.polyval(pl, y_end))
    cols = np.arange(int(xc_b - 0.8 * r_b), int(xc_b + 0.8 * r_b) + 1)
    bottom = []
    for x in cols:
        yy = np.nonzero(m[:, x])[0]
        if len(yy):
            bottom.append((x, yy.max()))
    sag, e_b = 0.0, 0.0
    if len(bottom) > 10:
        bx, by = np.array(bottom, float).T
        q = np.polyfit((bx - xc_b) / r_b, by, 2)            # y = a t^2 + b t + c, t in [-0.8, 0.8]
        sag = -q[0]                                          # >0: centre lower than the sides (camera above)
        e_b = float(np.clip(sag / r_b, -0.6, 0.6))           # y(theta) = c - e R (1 - cos)  ~  c - e R t^2 / 2 ...
        e_b = float(np.clip(2 * sag / r_b, -0.8, 0.8))       # ... so e = 2 a / R for small angles
    curved_base = abs(sag) > 0.02 * r_b
    info = {"body_rows": [int(yb[0]), int(yb[-1])], "left": pl.tolist(), "right": pr.tolist(),
            "side_residual_px": round(float(res), 2), "width_px": round(float(wmax), 1),
            "base_sagitta_px": round(float(sag), 2), "e_base": round(e_b, 4), "y_base": y_end}
    if straight and parallel and curved_base:
        info.update(kind="round", reason="straight parallel sides and a curved base")
    elif straight and parallel:
        info.update(kind="round_or_flat", reason="straight parallel sides, straight base (eye level)")
    else:
        info.update(kind="flat", reason="sides not straight or not parallel")
    return info


def fit_quad(matte_ref):
    """Four-sided outline of the surface (perspective rectangle). Returns (4,2) TL,TR,BR,BL and the IoU with the matte."""
    m = matte_ref.astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None, 0.0
    c = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in np.linspace(0.01, 0.12, 23):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2).astype(float)
            break
    if quad is None:
        return None, 0.0
    qm = np.zeros_like(m)
    cv2.fillConvexPoly(qm, quad.astype(np.int32), 1)
    iou = float((qm & m).sum()) / float((qm | m).sum() + 1)
    return order_quad(quad), iou


def order_quad(quad):
    """4 points -> TL, TR, BR, BL: clockwise around the centre, starting at the top edge's left end."""
    q = np.asarray(quad, float)
    c = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]   # clockwise on screen (y points down)
    top = min(range(4), key=lambda i: (q[i, 1] + q[(i + 1) % 4, 1]))
    q = np.roll(q, -top, axis=0)
    if q[0, 0] > q[1, 0]:  # the top edge must run left to right
        q = q[[1, 0, 3, 2]]
    return q


def rect_aspect(quad, w, h, f=None):
    """Width/height of the real rectangle seen as `quad` (TL,TR,BR,BL) by a pinhole camera with the principal point
    at the centre (Zhang & He 2007). f is estimated from the vanishing points when the edges aren't parallel."""
    u0, v0 = w / 2.0, h / 2.0
    m1, m2, m4, m3 = [np.array([p[0], p[1], 1.0]) for p in quad]  # m1 TL (0,0), m2 TR (W,0), m3 BL (0,H), m4 BR
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    n2, n3 = k2 * m2 - m1, k3 * m3 - m1
    f_est = None
    if abs(n2[2] * n3[2]) > 1e-9:
        f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0)
               + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)) / (n2[2] * n3[2])
        if f2 > 0:
            f_est = math.sqrt(f2)
    big = max(w, h)
    if f is None:
        f = f_est if (f_est is not None and 0.5 * big < f_est < 4 * big) else 1.2 * big
    K = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1.0]])
    Ki = np.linalg.inv(K)
    A = Ki.T @ Ki
    ar = math.sqrt(max(1e-9, (n2 @ A @ n2) / max(1e-9, n3 @ A @ n3)))
    return ar, f


class Layout:
    """Maps texture coordinates (u, v) of the text plane to the reference frame (x, y) at full resolution.

    kind 'plane': (x, y) = G (u, v) (homography; G = identity when the text faces the reference view).
    kind 'round': cylinder with axis xc(y) and radius R(y) (lines in y); u is arc length at radius R0,
      x = xc(v') + R(v') sin(u / R0), y = v - e(v) R(v) (1 - cos(u / R0)), e(y) = (y - y_h) / f."""

    def __init__(self, kind, G=None, cyl=None):
        self.kind = kind
        self.G = G if G is not None else np.eye(3)
        self.Gi = np.linalg.inv(self.G)
        self.cyl = cyl

    def tex_to_ref(self, u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        if self.kind == "plane":
            p = np.stack([u, v, np.ones_like(u)], -1) @ self.G.T
            return p[..., 0] / p[..., 2], p[..., 1] / p[..., 2]
        c = self.cyl
        th = u / c["R0"]
        R = c["r1"] * v + c["r0"]
        xc = c["c1"] * v + c["c0"]
        e = (v - c["y_h"]) / c["f"]
        return xc + R * np.sin(th), v - e * R * (1 - np.cos(th))

    def ref_to_tex(self, x, y):
        """Returns u, v and a validity mask (False behind a cylinder's silhouette)."""
        if self.kind == "plane":
            p = np.stack([x, y, np.ones_like(x)], -1) @ self.Gi.T
            return p[..., 0] / p[..., 2], p[..., 1] / p[..., 2], np.ones(np.shape(x), bool)
        c = self.cyl
        v = np.array(y, float, copy=True)
        for _ in range(3):
            R = c["r1"] * v + c["r0"]
            xc = c["c1"] * v + c["c0"]
            s = (x - xc) / np.maximum(R, 1e-3)
            sc = np.clip(s, -1, 1)
            th = np.arcsin(sc)
            e = (v - c["y_h"]) / c["f"]
            v = y + e * R * (1 - np.cos(th))
        return th * c["R0"], v, np.abs(s) < 0.985

    def scale_at(self, u, v):
        """Approximate ref px per texture unit around (u, v)."""
        x0, y0 = self.tex_to_ref(np.array([u, u + 1, u]), np.array([v, v, v + 1]))
        return math.sqrt(abs((x0[1] - x0[0]) * (y0[2] - y0[0]) - (x0[2] - x0[0]) * (y0[1] - y0[0])))


def _inscribed_rect(mask, frac_target=0.985):
    """Large axis-aligned rectangle inside a binary mask (shrinks the bounding box from its worst side)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    integ = cv2.integral(mask.astype(np.uint8))

    def fill(a, b, c, d):
        s = integ[d + 1, b + 1] - integ[c, b + 1] - integ[d + 1, a] + integ[c, a]
        return s / max(1, (b - a + 1) * (d - c + 1))

    for _ in range(4000):
        if x1 - x0 < 4 or y1 - y0 < 4:
            break
        if fill(x0, x1, y0, y1) >= frac_target:
            break
        cand = [("l", fill(x0, x0, y0, y1)), ("r", fill(x1, x1, y0, y1)), ("t", fill(x0, x1, y0, y0)),
                ("b", fill(x0, x1, y1, y1))]
        side = min(cand, key=lambda z: z[1])[0]
        if side == "l":
            x0 += 1
        elif side == "r":
            x1 -= 1
        elif side == "t":
            y0 += 1
        else:
            y1 -= 1
    return x0, y0, x1, y1


def build_layout(matte_ref, frame_shape, shape="Auto", quad=None, log=None):
    """Chooses the surface model from the reference matte (full res bool). Returns (Layout, usable box (u0,v0,u1,v1)
    in texture units, report dict)."""
    log = log or (lambda *a: None)
    h, w = frame_shape
    rep = {}
    rnd = fit_round(matte_ref) if shape in ("Auto", "Round") else {"kind": "flat", "reason": "set to Flat"}
    rep["shape_test"] = rnd
    want_round = shape == "Round" or (shape == "Auto" and rnd.get("kind") == "round")
    if shape == "Round" and rnd.get("kind") in ("none", "flat") and "left" not in rnd:
        want_round = False
    if want_round:
        y0b, y1b = rnd["body_rows"]
        pl, pr = rnd["left"], rnd["right"]
        c1, c0 = 0.5 * (pl[0] + pr[0]), 0.5 * (pl[1] + pr[1])
        r1, r0 = 0.5 * (pr[0] - pl[0]), 0.5 * (pr[1] - pl[1])
        f = 1.2 * max(w, h)
        y_base = rnd["y_base"]
        e_b = rnd["e_base"]
        y_h = y_base - e_b * f
        ym = 0.5 * (y0b + y1b)
        R0 = r1 * ym + r0
        cyl = {"c0": c0, "c1": c1, "r0": r0, "r1": r1, "R0": R0, "f": f, "y_h": y_h}
        lay = Layout("round", cyl=cyl)
        th_max = math.radians(55)
        pad = 0.06 * (y1b - y0b)
        usable = (-th_max * R0, y0b + pad, th_max * R0, y1b - pad)
        rep.update(surface="Round", cylinder={"radius_px": round(R0, 1), "horizon_y": round(y_h, 1),
                                               "e_base": e_b, "body_rows": [y0b, y1b]})
        return lay, usable, rep
    if quad is None:
        quad, iou = fit_quad(matte_ref)
    else:
        iou = 1.0
    if quad is not None and iou >= 0.88:
        ar, f = rect_aspect(quad, w, h)
        Hq = float(np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
        Wq = ar * Hq
        src = np.float32([[0, 0], [Wq, 0], [Wq, Hq], [0, Hq]])
        G = cv2.getPerspectiveTransform(src, quad.astype(np.float32)).astype(np.float64)
        lay = Layout("plane", G=G)
        inset = 0.07
        usable = (inset * Wq, inset * Hq, (1 - inset) * Wq, (1 - inset) * Hq)
        rep.update(surface="Flat (perspective)", quad=quad.round(1).tolist(), quad_iou=round(iou, 3),
                   rect_aspect=round(ar, 3), focal_px=round(f, 1))
        return lay, usable, rep
    r = _inscribed_rect(cv2.erode(matte_ref.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    x0, y0, x1, y1 = r
    inset = 0.05
    dx, dy = inset * (x1 - x0), inset * (y1 - y0)
    rep.update(surface="Flat (facing the reference view)", quad_iou=round(iou, 3))
    return Layout("plane"), (x0 + dx, y0 + dy, x1 - dx, y1 - dy), rep


# ------------------------------------------------------------------------------------------------- text image
def _script(text):
    """The first non-Latin script found in the text (Japanese wins over Chinese when kana are present)."""
    cps = [ord(c) for c in text]
    for name, ranges, f, shaping in SCRIPT_FONTS:
        if any(lo <= cp <= hi for cp in cps for lo, hi in ranges):
            return name, f, shaping
    return None


def resolve_font(font="Montserrat", bold=False, font_file=None, text=""):
    """Font spec (path, weight on a variable font or None), or None for Pillow's built-in font; plus notes."""
    notes = []
    if font_file:
        if os.path.isfile(font_file):
            return (font_file, None), notes
        raise ValueError(f"Font file not found: {font_file}")
    if os.path.isfile(str(font)):
        return (str(font), None), notes
    sc = _script(text or "")
    if sc:
        name, f, shaping = sc
        p = os.path.join(FONT_DIR, f)
        if shaping:
            try:
                from PIL import features
                if not features.check("raqm"):
                    notes.append(f"{name} text needs Pillow with raqm for joined letters and direction: check the result")
            except Exception:
                pass
        if os.path.isfile(p):
            notes.append(f"{name} text: set in {f.split('[')[0]} (the chosen font has no {name} letters)")
            return (p, 700 if bold else 400), notes
        notes.append(f"{name} text but its Noto font isn't installed with the node: letters may be missing")
    spec = FONTS.get(font)
    if spec:
        f, w = spec["bold" if bold else "regular"]
        p = os.path.join(FONT_DIR, f)
        if os.path.isfile(p):
            return (p, w), notes
    notes.append(f"Font '{font}' isn't installed with the node: a fallback font was used")
    return None, notes


def _font(spec, px):
    """spec: None (Pillow's built-in font), a path, or (path, weight on a variable font or None)."""
    px = max(4, int(round(px)))
    if not spec:
        return ImageFont.load_default(size=px)
    path, wght = (spec, None) if isinstance(spec, str) else spec
    font = ImageFont.truetype(path, px)
    if wght is not None:
        try:
            axes = font.get_variation_axes()
            vals = []
            for ax in axes:
                nm = ax.get("name", b"")
                nm = nm.decode("latin-1", "ignore") if isinstance(nm, bytes) else str(nm)
                vals.append(min(max(wght, ax["minimum"]), ax["maximum"]) if nm.lower().startswith("weight")
                            else ax["default"])
            font.set_variation_by_axes(vals)
        except Exception:
            pass  # not a variable font, or FreeType without variations: its default weight is used
    return font


def render_text(text, font_path, px, align="center", line_spacing=1.18, letter_spacing=0.0):
    """Text block as a float32 alpha image at `px` font size, tightly cropped to the ink + 2 px.
    Returns alpha and metrics (cap height in px)."""
    font = _font(font_path, px)
    lines = text.replace("\\n", "\n").split("\n")
    asc, desc = font.getmetrics()
    lh = int(round((asc + desc) * line_spacing))
    widths = []
    for ln in lines:
        if letter_spacing:
            wsum = sum(font.getlength(ch) for ch in ln) + letter_spacing * px * max(0, len(ln) - 1)
        else:
            wsum = font.getlength(ln)
        widths.append(wsum)
    W = int(math.ceil(max(widths) if widths else 1)) + 8
    Hh = lh * len(lines) + desc + 8
    img = Image.new("L", (W, Hh), 0)
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        x = {"left": 4, "right": W - 4 - widths[i]}.get(align, (W - widths[i]) / 2)
        y = 4 + i * lh
        if letter_spacing:
            for ch in ln:
                d.text((x, y), ch, font=font, fill=255)
                x += font.getlength(ch) + letter_spacing * px
        else:
            d.text((x, y), ln, font=font, fill=255)
    a = np.asarray(img, np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.01)
    if len(xs) == 0:
        return np.zeros((4, 4), np.float32), {"cap_px": 0.0}
    a = a[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3]
    try:
        cap = font.getbbox("H")
        cap_px = float(cap[3] - cap[1])
    except Exception:
        cap_px = 0.7 * px
    return a, {"cap_px": cap_px, "lines": len(lines)}


def _detail_map(gray_ref, layout, usable, step):
    """Local detail (gradient energy) sampled on the usable region in texture units: for Auto placement."""
    u0, v0, u1, v1 = usable
    us = np.arange(u0, u1, step)
    vs = np.arange(v0, v1, step)
    U, V = np.meshgrid(us, vs)
    x, y = layout.tex_to_ref(U, V)
    g = cv2.GaussianBlur(gray_ref.astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    e = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2.0)
    return cv2.remap(e, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR), us, vs


def place_text(text, font_path, layout, usable, placement="Center", size="Auto", gray_ref=None, align=None):
    """Font size and the text box (u0, v0, u1, v1) in texture units."""
    u0, v0, u1, v1 = usable
    UW, UH = u1 - u0, v1 - v0
    pl = "Auto" if placement in ("Auto", "Emptiest Area") else placement if placement in PLACEMENTS else "Center"
    if align is None:
        align = "left" if "Left" in pl else "right" if "Right" in pl else "center"
    a, met = render_text(text, font_path, 200, align)
    bw, bh = a.shape[1], a.shape[0]
    frac = SIZES.get(size)
    max_h = UH * (0.45 if pl in ("Center", "Left", "Right", "Auto") else 0.3)
    if frac is None:
        frac = 0.75
    scale = min(frac * UW / bw, max_h / bh)
    px = 200 * scale
    tw, th = bw * scale, bh * scale
    mu, mv = 0.04 * UW, 0.04 * UH
    if pl == "Auto" and gray_ref is not None:
        step = max(1.0, min(UW, UH) / 60)
        e, us, vs = _detail_map(gray_ref, layout, usable, step)
        kw, kh = max(1, int(round(tw / step))), max(1, int(round(th / step)))
        integ = cv2.integral(e.astype(np.float64))
        best, bu, bv = None, 0, 0
        H_, W_ = e.shape
        for j in range(0, max(1, H_ - kh + 1)):
            for i in range(0, max(1, W_ - kw + 1)):
                s = integ[j + kh, i + kw] - integ[j, i + kw] - integ[j + kh, i] + integ[j, i]
                cu = (i + kw / 2) / max(1, W_)
                cv = (j + kh / 2) / max(1, H_)
                s = s / (kw * kh) * (1 + 0.6 * ((cu - 0.5) ** 2 + (cv - 0.5) ** 2))  # mild centre preference
                if best is None or s < best:
                    best, bu, bv = s, i, j
        cu = u0 + bu * step + tw / 2
        cv = v0 + bv * step + th / 2
    else:
        cu = {"Left": u0 + mu + tw / 2, "Right": u1 - mu - tw / 2}.get(
            "Left" if "Left" in pl else "Right" if "Right" in pl else "", (u0 + u1) / 2)
        cv = (v0 + mv + th / 2) if pl.startswith("Top") else (v1 - mv - th / 2) if pl.startswith("Bottom") else (v0 + v1) / 2
    return {"px": px, "box": (cu - tw / 2, cv - th / 2, cu + tw / 2, cv + th / 2), "align": align,
            "cap_tex": met["cap_px"] * scale, "placement": pl}


class TextTexture:
    """Supersampled text alpha in texture units with mip levels."""

    def __init__(self, text, font_path, place, texels_per_unit, align):
        self.k = texels_per_unit
        a, _ = render_text(text, font_path, place["px"] * texels_per_unit, align)
        u0, v0, u1, v1 = place["box"]
        # rendered block (cropped to ink) mapped exactly onto the box
        tw, th = (u1 - u0) * self.k, (v1 - v0) * self.k
        a = cv2.resize(a, (max(2, int(round(tw))), max(2, int(round(th)))), interpolation=cv2.INTER_AREA)
        self.pad = 4
        a = cv2.copyMakeBorder(a, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_CONSTANT, value=0)
        self.box = place["box"]
        self.levels = [a]
        while min(self.levels[-1].shape) > 16:
            self.levels.append(cv2.pyrDown(self.levels[-1]))

    def coords(self, u, v, level):
        """Texture coordinates -> pixel-centre coordinates of mip level `level` (OpenCV convention)."""
        t0x = (u - self.box[0]) * self.k - 0.5 + self.pad
        t0y = (v - self.box[1]) * self.k - 0.5 + self.pad
        f = 2.0 ** level
        return (t0x - (f - 1) / 2) / f, (t0y - (f - 1) / 2) / f


# --------------------------------------------------------------------------------------------- per-frame draw
def _box_outline(box, n=24):
    u0, v0, u1, v1 = box
    t = np.linspace(0, 1, n)
    us = np.concatenate([u0 + (u1 - u0) * t, np.full(n, u1), u1 - (u1 - u0) * t, np.full(n, u0)])
    vs = np.concatenate([np.full(n, v0), v0 + (v1 - v0) * t, np.full(n, v1), v1 - (v1 - v0) * t])
    return us, vs


def text_alpha(tex, layout, Hs, weights, frame_hw, ss=3):
    """Alpha of the text in one output frame. Hs: list of ref->frame homographies (full res) over the shutter,
    weights: their weights. Returns (alpha crop, x0, y0) or None when the text is out of frame."""
    h, w = frame_hw
    us, vs = _box_outline(tex.box)
    xr, yr = layout.tex_to_ref(us, vs)
    pts = []
    for H in Hs:
        pts.append(_apply_h(H, np.stack([xr, yr], 1)))
    pts = np.concatenate(pts)
    x0, y0 = int(math.floor(pts[:, 0].min())) - 2, int(math.floor(pts[:, 1].min())) - 2
    x1, y1 = int(math.ceil(pts[:, 0].max())) + 2, int(math.ceil(pts[:, 1].max())) + 2
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    bw, bh = x1 - x0, y1 - y0
    if bw * bh * ss * ss > 24e6:
        ss = max(1, int(math.sqrt(24e6 / (bw * bh))))
    xs = x0 + (np.arange(bw * ss) + 0.5) / ss - 0.5
    ys = y0 + (np.arange(bh * ss) + 0.5) / ss - 0.5
    X, Y = np.meshgrid(xs, ys)
    acc = np.zeros((bh, bw), np.float32)
    for H, wt in zip(Hs, weights):
        Hi = np.linalg.inv(H)
        d = Hi[2, 0] * X + Hi[2, 1] * Y + Hi[2, 2]
        qx = (Hi[0, 0] * X + Hi[0, 1] * Y + Hi[0, 2]) / d
        qy = (Hi[1, 0] * X + Hi[1, 1] * Y + Hi[1, 2]) / d
        u, v, valid = layout.ref_to_tex(qx, qy)
        # mip level from the texel footprint of one sub-sample
        cy, cx = u.shape[0] // 2, u.shape[1] // 2
        cx1, cy1 = min(cx + 1, u.shape[1] - 1), min(cy + 1, u.shape[0] - 1)
        du = math.hypot(u[cy, cx1] - u[cy, cx], v[cy, cx1] - v[cy, cx]) * tex.k
        dv = math.hypot(u[cy1, cx] - u[cy, cx], v[cy1, cx] - v[cy, cx]) * tex.k
        foot = max(1e-6, math.sqrt(du * dv))
        level = int(np.clip(math.floor(math.log2(max(foot, 1.0))), 0, len(tex.levels) - 1))
        tx, ty = tex.coords(u, v, level)
        a = cv2.remap(tex.levels[level], tx.astype(np.float32), ty.astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = a * valid
        a = cv2.resize(a, (bw, bh), interpolation=cv2.INTER_AREA) if ss > 1 else a
        acc += wt * a
    return np.clip(acc, 0, 1), x0, y0


def motion_paths(H, quad_ref):
    """Image paths of 4 reference points and their per-frame velocity (5-frame Savitzky-Golay derivative, with
    single-frame spikes clamped so one bad frame can't smear the text)."""
    n = len(H)
    P = np.stack([_apply_h(H[t], quad_ref) for t in range(n)])
    V = np.zeros_like(P)
    for t in range(n):
        if 2 <= t < n - 2:
            V[t] = (-2 * P[t - 2] - P[t - 1] + P[t + 1] + 2 * P[t + 2]) / 10
        elif 1 <= t < n - 1:
            V[t] = (P[t + 1] - P[t - 1]) / 2
        elif n > 1:
            V[t] = P[1] - P[0] if t == 0 else P[-1] - P[-2]
    mag = np.linalg.norm(V, axis=2).max(1)
    for t in range(n):
        med = float(np.median(mag[max(0, t - 3):t + 4]))
        if mag[t] > 2 * med + 1:
            V[t] *= (2 * med + 1) / mag[t]
    return P, V


def blur_set(H, t, quad_ref, P, V, blur=True):
    """Shutter samples for frame t (homographies + weights) and the blur length in px."""
    length = float(np.linalg.norm(V[t], axis=1).max()) * SHUTTER
    if not blur or length < 1.0:   # under a pixel: invisible, skip
        return [H[t]], [1.0], length
    k = int(min(24, math.ceil(length / 0.5)))
    taus = (np.arange(k) + 0.5) / k * SHUTTER - SHUTTER / 2
    q = quad_ref.astype(np.float32)
    Hs = [cv2.getPerspectiveTransform(q, (P[t] + tau * V[t]).astype(np.float32)).astype(np.float64) for tau in taus]
    return Hs, [1.0 / k] * k, length


# ------------------------------------------------------------------------------------------------- the look
class Shader:
    """Light of the surface under the text: shading relative to the clip's typical level."""

    def __init__(self, cap_px_screen):
        self.sig = max(1.5, 0.35 * cap_px_screen)
        self.k_close = _odd(max(3, 0.3 * cap_px_screen))

    def light(self, rgb_lin, w, light_surface=True):
        """rgb_lin (h,w,3) linear crop, w (h,w) surface weight. Returns luminance light S, chroma C (h,w,3), detail D."""
        L = _lum(rgb_lin).astype(np.float32)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.k_close, self.k_close))
        Lc = cv2.morphologyEx(L, cv2.MORPH_CLOSE if light_surface else cv2.MORPH_OPEN, ker)
        S = _blur_w(Lc, w, self.sig)
        C = _blur_w(rgb_lin, w, 2 * self.sig) / np.maximum(_blur_w(L, w, 2 * self.sig), 1e-5)[..., None]
        D = Lc / np.maximum(cv2.GaussianBlur(Lc, (0, 0), 2.0), 1e-5)
        D = np.clip(D, 0.85, 1.18) ** 0.6
        return S, C, D


def shade_margin(shader):
    return int(math.ceil(3 * shader.sig)) + shader.k_close


def composite(region_rgb, alpha, ax, ay, ink_lin, vis_region, shader, S_ref, C_ref, white_cap=None, light_surface=True):
    """region_rgb: (h,w,3) float sRGB crop of the frame around the text (text alpha at offset ax, ay inside it, plus a
    margin for the light estimate). vis_region: (h,w) visible surface. Returns the new pixels under the alpha box
    (float32 sRGB; exact original values where the text doesn't reach) and the effective alpha."""
    bh, bw = alpha.shape
    lin = srgb_to_lin(region_rgb[..., :3])
    wv = vis_region.astype(np.float32)
    S, C, D = shader.light(lin, np.maximum(wv, 1e-3), light_surface)
    sl = (slice(ay, ay + bh), slice(ax, ax + bw))
    a = alpha * wv[sl]
    shade = np.clip(S[sl] / max(S_ref, 1e-5), 0.06, 3.0)[..., None] * (C[sl] / np.maximum(C_ref, 1e-5)) * D[sl][..., None]
    ink = np.asarray(ink_lin, np.float32)
    if white_cap is not None and _lum(ink) > white_cap:
        ink = ink * (white_cap / _lum(ink))
    out_lin = lin[sl] * (1 - a[..., None]) + ink[None, None, :] * shade * a[..., None]
    orig = region_rgb[sl][..., :3].astype(np.float32)
    return np.where((a > 0)[..., None], lin_to_srgb(out_lin), orig), a


def apply_patch(frame, patch):
    """Pastes a patch from text_on_surface into a copy of the frame (uint8 or float)."""
    out = frame.copy()
    if patch is None:
        return out
    y0, x0, rgb = patch["y0"], patch["x0"], patch["rgb"]
    h, w = rgb.shape[:2]
    if out.dtype == np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
        a = patch["alpha"] > 0
        out[y0:y0 + h, x0:x0 + w, :3] = np.where(a[..., None], rgb, out[y0:y0 + h, x0:x0 + w, :3])
    else:
        out[y0:y0 + h, x0:x0 + w, :3] = rgb.astype(out.dtype)
    return out


# ------------------------------------------------------------------------------------- appearance occlusion
def _guided_np(guide, p, r, eps=1e-3):
    """Guided filter (He et al.): p smoothed along the guide's edges."""
    k = (2 * r + 1, 2 * r + 1)
    mi, mp = cv2.blur(guide, k), cv2.blur(p, k)
    a = (cv2.blur(guide * p, k) - mi * mp) / (cv2.blur(guide * guide, k) - mi * mi + eps)
    b = mp - a * mi
    return cv2.blur(a, k) * guide + cv2.blur(b, k)


class OcclusionCheck:
    """Things in front of the text that the matte doesn't hold (or no matte at all): each frame is warped back onto
    the reference frame and compared with it around the text, by colour (shading-invariant chromaticity), local
    structure (normalised cross-correlation) and a brightness ratio beyond what light changes do. A per-pixel
    baseline over the clip removes what always differs (sparkle, view-dependent reflections, generated-footage
    drift); an occluder must also be there in a neighbouring frame and be at least about half a letter in size.
    Edges are snapped to the frame's own edges (guided filter). Masks live in reference coordinates."""

    def __init__(self, ref_rgb, layout, box, cap_ref, frame_hw, surface_ref=None):
        H, W = frame_hw
        us, vs = _box_outline(box)
        xr, yr = layout.tex_to_ref(us, vs)
        mx = 0.15 * (xr.max() - xr.min()) + 6
        my = 0.4 * (yr.max() - yr.min()) + 6
        self.x0, self.y0 = int(max(0, xr.min() - mx)), int(max(0, yr.min() - my))
        self.x1, self.y1 = int(min(W, xr.max() + mx)), int(min(H, yr.max() + my))
        self.T = np.array([[1, 0, self.x0], [0, 1, self.y0], [0, 0, 1.0]])
        self.size = (self.x1 - self.x0, self.y1 - self.y0)
        self.R = ref_rgb[self.y0:self.y1, self.x0:self.x1, :3].astype(np.float32)
        self.cap = max(4.0, cap_ref)
        self.frame_hw = frame_hw
        # only near the text and on the surface: that's where an occluder matters
        near = np.zeros((self.y1 - self.y0, self.x1 - self.x0), np.uint8)
        poly = np.stack([xr - self.x0, yr - self.y0], 1)
        cv2.fillPoly(near, [np.round(poly).astype(np.int32)], 1)
        r = _odd(1.0 * self.cap)
        near = cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
        if surface_ref is not None:
            near &= surface_ref[self.y0:self.y1, self.x0:self.x1].astype(np.uint8)
        self.near = near > 0
        self.sig = max(1.2, 0.12 * self.cap)

    def _score(self, frame_rgb, Hf, zoom):
        Hc = Hf @ self.T
        warped = cv2.warpPerspective(frame_rgb[..., :3].astype(np.float32), Hc, self.size,
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(-1, -1, -1))
        inside = warped[..., 0] >= 0
        if inside.sum() < 50:
            return None, inside, None
        R = self.R
        Fw = np.where(inside[..., None], warped, R)
        sig = max(self.sig, 0.8 / max(zoom, 0.3))
        Rb, Fb = cv2.GaussianBlur(R, (0, 0), sig), cv2.GaussianBlur(Fw, (0, 0), sig)
        # sparkle (small highlights that come and go with the view) in either image: not evidence
        lr0, lf0 = _lum(R), _lum(Fw)
        med_r = cv2.medianBlur((np.clip(lr0, 0, 1) * 255).astype(np.uint8), _odd(max(5, 0.6 * self.cap))) / 255.0
        med_f = cv2.medianBlur((np.clip(lf0, 0, 1) * 255).astype(np.uint8), _odd(max(5, 0.6 * self.cap))) / 255.0
        sparkle = cv2.dilate(((lr0 > med_r * 1.25 + 0.04) | (lf0 > med_f * 1.25 + 0.04)).astype(np.uint8),
                             np.ones((3, 3), np.uint8)) > 0
        rc = Rb / (Rb.sum(-1, keepdims=True) + 0.02)
        fc = Fb / (Fb.sum(-1, keepdims=True) + 0.02)
        shift = np.median((fc - rc)[inside], axis=0)
        dchroma = np.linalg.norm(fc - rc - shift, axis=-1)
        lr, lf = _lum(Rb), _lum(Fb)
        k = _odd(max(5, 6 * sig))
        mr, mf = cv2.blur(lr, (k, k)), cv2.blur(lf, (k, k))
        vr = cv2.blur(lr * lr, (k, k)) - mr * mr
        vf = cv2.blur(lf * lf, (k, k)) - mf * mf
        ncc = (cv2.blur(lr * lf, (k, k)) - mr * mf) / np.sqrt(np.maximum(vr * vf, 1e-10))
        textured = np.maximum(vr, vf) > 1.2e-3  # faint glows and grain are not texture
        ratio = (lf + 0.02) / (lr + 0.02)
        dev = np.abs(np.log(ratio / np.median(ratio[inside])))
        # reflections: brighter than the surface's usual level in either image -> brightness alone isn't evidence
        nr = self.near & inside
        if nr.any():
            hi_r = lr > 1.15 * np.percentile(lr[nr], 85)
            hi_f = lf > 1.15 * np.percentile(lf[nr], 85)
        else:
            hi_r = hi_f = np.zeros_like(lr, bool)
        hl = hi_r | hi_f
        score = np.where(hl, 0.6, 1.0) * dchroma / 0.035 + np.where(textured & ~hl, np.clip((0.55 - ncc) / 0.35, 0, None), 0) \
            + np.where(hl, 0, np.clip((dev - 0.7) / 0.4, 0, None))
        score = np.where(sparkle | ~self.near, 0, score)
        return score.astype(np.float32), inside & self.near, _lum(Fw).astype(np.float32)

    def run(self, get_frame, Hf, zoom, n, ref, log=None):
        scores, insides, guides = [], [], []
        for t in range(n):
            if t == ref:
                scores.append(None)
                insides.append(None)
                guides.append(None)
                continue
            sc, ins, g = self._score(get_frame(t), Hf[t], zoom[t])
            scores.append(None if sc is None else sc.astype(np.float16))
            insides.append(ins)
            guides.append(None if g is None else g.astype(np.float16))
        valid = [i for i in range(n) if scores[i] is not None]
        if not valid:
            return [None] * n, {}
        stack = np.stack([np.where(insides[i], scores[i].astype(np.float32), np.nan) for i in valid], 0)
        base = np.nanmedian(stack, axis=0)
        base = np.where(np.isnan(base), 0, base)
        # motion: warped onto the reference, the surface stands still; whatever moves there is in front of it
        lum_ref = _lum(self.R)
        def lum8(i):
            g = lum_ref if i == ref else guides[i].astype(np.float32)
            return (np.clip(g, 0, 1) * 255).astype(np.uint8)
        raw = {}
        for j, i in enumerate(valid):
            app = ((stack[j] - base) > 1.0) & (stack[j] > 1.3)
            mot = np.zeros_like(app)
            for k_ in (i - 1, i + 1):
                if 0 <= k_ < n and (k_ == ref or guides[k_] is not None):
                    fl = cv2.calcOpticalFlowFarneback(lum8(k_), lum8(i), None, 0.5, 3,
                                                      _odd(max(9, 0.5 * self.cap)), 3, 5, 1.2, 0)
                    mot |= np.linalg.norm(fl, axis=2) > max(1.5, 0.06 * self.cap)
            raw[i] = (app | (mot & ((stack[j] - base) > 0.25))) & insides[i]
        del stack
        min_area = self.cap ** 2
        nb_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(0.6 * self.cap),) * 2)   # fast occluders move
        close_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(0.35 * self.cap),) * 2)
        masks = [None] * n
        stats = {"frames_flagged": 0, "flagged_share_max": 0.0}
        for i in valid:
            o = raw[i].astype(np.uint8)
            nb = np.zeros_like(o)
            for j in (i - 1, i + 1):
                if j in raw:
                    nb |= raw[j].astype(np.uint8)
            o &= cv2.dilate(nb, nb_ker)
            o = cv2.morphologyEx(o, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            o = cv2.morphologyEx(o, cv2.MORPH_CLOSE, close_ker)
            # an occluder's inside can look like the surface (a cream wing over a cream label): fill enclosed holes
            if o.any():
                ff = np.pad(o, 1)
                cv2.floodFill(ff, None, (0, 0), 2)
                o = (o | (ff[1:-1, 1:-1] == 0)).astype(np.uint8)
            num, lab, st, _ = cv2.connectedComponentsWithStats(o, 8)
            keep = np.zeros_like(o)
            for c in range(1, num):
                if st[c, cv2.CC_STAT_AREA] >= min_area:
                    keep[lab == c] = 1
            if not keep.any():
                continue
            keep = cv2.dilate(keep, np.ones((3, 3), np.uint8)).astype(np.float32)
            g = guides[i].astype(np.float32)
            g = g / (g.max() + 1e-6)
            soft = np.clip(_guided_np(g, keep, max(2, int(round(0.08 * self.cap))), 1e-3), 0, 1)
            soft = soft * insides[i]
            masks[i] = soft.astype(np.float32)
            stats["frames_flagged"] += 1
            stats["flagged_share_max"] = max(stats["flagged_share_max"], float((soft > 0.5).mean()))
        return masks, stats

    def in_frame(self, mask_ref, Hf, x0, y0, w, h):
        """Occlusion mask (reference crop) -> frame region (x0, y0, w, h)."""
        Tb = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
        M = Tb @ Hf @ self.T
        return cv2.warpPerspective(mask_ref, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)


# ------------------------------------------------------------------------------------------------- the edit
def choose_reference(mattes_small, grays):
    """Frame with the most visible surface (not cut by the frame edge), ties broken by sharpness."""
    best, bi = -1.0, 0
    for i, (m, g) in enumerate(zip(mattes_small, grays)):
        mb = m > 0.5
        area = float(mb.sum())
        if area < 50:
            continue
        edge = mb[0].sum() + mb[-1].sum() + mb[:, 0].sum() + mb[:, -1].sum()
        lap = cv2.Laplacian(g, cv2.CV_32F)
        sharp = float(np.sqrt((lap[mb] ** 2).mean())) if mb.any() else 0.0
        score = area * (0.6 if edge > 0 else 1.0) * (1 + 0.02 * min(sharp, 25))
        if score > best:
            best, bi = score, i
    return bi


def text_on_surface(frames, text, mattes=None, placement="Center", size="Auto", font="Montserrat", bold=False,
                    font_file=None, colour="", shape="Auto", motion_blur=True, occlusion_check=False,
                    ref=None, surface_ref=None, quad=None, occluders=None, log=None):
    """frames: list/array of (H,W,3) float sRGB in [0,1] (or uint8). mattes: per-frame (H,W) surface mattes (float),
    or None with `surface_ref` (bool mask of the surface in frame `ref`, e.g. a drawn polygon). occluders: optional
    per-frame (H,W) masks of things in front. Returns (patches, report): per frame None (no text) or a dict with
    y0, x0, rgb (new pixels, float sRGB) and alpha (the text's coverage) - paste with apply_patch()."""
    log = log or (lambda *a: None)
    n = len(frames)
    Hh, Ww = frames[0].shape[:2]
    f32 = lambda im: im.astype(np.float32) / 255.0 if im.dtype == np.uint8 else im.astype(np.float32)
    s = min(1.0, ANALYSIS_LONG / max(Hh, Ww))
    aw, ah = int(round(Ww * s)), int(round(Hh * s))
    grays = [cv2.cvtColor((np.clip(f32(f), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for f in frames]
    grays = [cv2.resize(g, (aw, ah), interpolation=cv2.INTER_AREA) for g in grays]
    rep = {"edit": "text_on_surface", "text": text, "placement": placement, "size": size, "font": font_file or font,
           "colour_input": colour, "frames": n, "analysis_scale": round(s, 4), "notes": [], "review_reasons": []}
    auto_colour = not (colour or "").strip() or str(colour).strip().lower() == "auto"
    if not auto_colour:
        rgb_t, hex_t = parse_colour(colour)
        rep["colour"] = hex_t
        ink_lin = srgb_to_lin(np.array(rgb_t, np.float32) / 255.0)

    small = None
    if mattes is not None:
        small = [cv2.resize(np.asarray(m, np.float32), (aw, ah), interpolation=cv2.INTER_AREA) for m in mattes]
    occ_small = None
    if occluders is not None:
        occ_small = [cv2.resize(np.asarray(m, np.float32), (aw, ah), interpolation=cv2.INTER_AREA) for m in occluders]
    if ref is None:
        ref = choose_reference(small, grays) if small is not None else 0
    rep["reference_frame"] = int(ref)
    if surface_ref is not None:
        surf_full = np.asarray(surface_ref).astype(bool)
    else:
        surf_full = np.asarray(mattes[ref]) > 0.5
    if surf_full.sum() < 100:
        rep["review_reasons"].append("The surface is not visible in the reference frame: no text was added")
        rep["needs_review"] = True
        return [None] * n, rep

    # -- layout on the reference frame (full res) -----------------------------------------------------------
    font_path, fnotes = resolve_font(font, bold, font_file, text)
    for nt in fnotes:
        (rep["review_reasons"] if ("missing" in nt or "fallback" in nt or "check" in nt) else rep["notes"]).append(nt)
    lay, usable, lrep = build_layout(surf_full, (Hh, Ww), shape, quad=quad, log=log)
    rep["layout"] = lrep
    gref = cv2.cvtColor((np.clip(f32(frames[ref]), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    place = place_text(text, font_path, lay, usable, placement, size, gray_ref=gref)
    if lay.kind == "round":
        # wrap only when it is visible: planar vs cylinder deviation at the text's edge
        th = max(abs(place["box"][0]), abs(place["box"][2])) / lay.cyl["R0"]
        dev = lay.cyl["R0"] * (th - math.sin(th))
        rep["layout"]["wrap_deviation_px"] = round(dev, 2)
        rep["layout"]["text_arc_deg"] = round(math.degrees(2 * th), 1)
    rep["text_box_tex"] = [round(v, 1) for v in place["box"]]
    rep["font_px_ref"] = round(place["px"], 1)

    # -- tracking ROI (analysis res): surface core; for a round surface, the front band of the body ----------
    roi_full = surf_full.copy()
    if lay.kind == "round":
        c = lay.cyl
        yy, xx = np.mgrid[0:Hh, 0:Ww]
        R = c["r1"] * yy + c["r0"]
        xc = c["c1"] * yy + c["c0"]
        y0b, y1b = lrep["cylinder"]["body_rows"]
        band = (np.abs(xx - xc) < 0.62 * R) & (yy >= y0b - 0.1 * (y1b - y0b)) & (yy <= y1b + 0.02 * (y1b - y0b))
        roi_full &= band
    roi = cv2.resize(roi_full.astype(np.uint8), (aw, ah), interpolation=cv2.INTER_NEAREST) > 0
    # a surface with too little detail of its own (plain wall, white marble) can't be tracked on its own: the region
    # grows into the surroundings until there is enough detail (the scene around a static surface moves with it)
    hp_ref = _highpass(grays[ref])
    detail = float(hp_ref[roi].std()) if roi.any() else 0.0
    rep["surface_detail"] = round(detail, 2)
    track_vis = small
    if detail < LOW_DETAIL:
        grown = roi.copy()
        for k in range(1, 9):
            r_ = _odd(k * 0.08 * max(aw, ah))
            grown = cv2.dilate(roi.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_, r_))) > 0
            if hp_ref[grown].std() >= LOW_DETAIL:
                break
        roi, track_vis = grown, None
        rep["tracking_region"] = "surface and its surroundings (the surface has little detail of its own)"
        rep["notes"].append("The surface has little detail of its own: it was tracked with its surroundings, "
                            "which is exact for a static surface and approximate for one that moves on its own")
    else:
        rep["tracking_region"] = "surface"
    tr = SurfaceTracker(grays, ref, roi, vis=track_vis, occ=occ_small, log=log)
    Ha, info = tr.run()
    Hf = np.stack([_ana_to_full(h_, aw / Ww, ah / Hh) for h_ in Ha])  # full-res ref -> frame
    rep["_H_raw"] = Hf.copy()

    # -- text texture --------------------------------------------------------------------------------------
    us, vs = _box_outline(place["box"], 8)
    xr, yr = lay.tex_to_ref(us, vs)
    ref_pts = np.stack([xr, yr], 1)
    quad_ref = ref_pts[[0, 8, 16, 24]]
    Hf = smooth_track(Hf, info, quad_ref, noise_px=0.35 / s)
    P_mo, V_mo = motion_paths(Hf, quad_ref)
    cu, cv = 0.5 * (place["box"][0] + place["box"][2]), 0.5 * (place["box"][1] + place["box"][3])
    k_ref = lay.scale_at(cu, cv)
    pc = ref_pts.mean(0)
    zoom = [_h_scale(Hf[t], pc) for t in range(n)]
    zmax = max(zoom) if zoom else 1.0
    k_tex = 3.0 * k_ref * zmax                      # ~3 texels per output pixel where the text is largest
    bw_u = place["box"][2] - place["box"][0]
    k_tex = max(0.05, min(k_tex, 8192.0 / max(bw_u, 1e-3), 2000.0 / max(place["px"], 1e-3)))
    tex = TextTexture(text, font_path, place, texels_per_unit=k_tex, align=place["align"])
    cap_screen = [place["cap_tex"] * k_ref * z for z in zoom]
    rep["cap_height_px"] = {"min": round(min(cap_screen), 1), "max": round(max(cap_screen), 1)}

    # -- shading reference (typical light over the clip) ----------------------------------------------------
    shader = Shader(float(np.median(cap_screen)))
    poly_ref = np.stack(lay.tex_to_ref(*_box_outline(place["box"], 16)), 1)
    S_list, C_list, P97 = [], [], []
    ref_f = f32(frames[ref])
    lum_ref_surface = float(_lum(srgb_to_lin(ref_f[..., :3]))[surf_full].mean())
    light_surface = lum_ref_surface > 0.12
    if auto_colour:  # automatic: contrasting ink (dark on a light surface, light on a dark one)
        hex_t = "#1E1E1E" if lum_ref_surface > 0.18 else "#F2F2F2"
        rgb_t, _ = parse_colour(hex_t)
        rep["colour"] = hex_t + " (automatic)"
        ink_lin = srgb_to_lin(np.array(rgb_t, np.float32) / 255.0)
    for t in range(0, n, 2):
        poly = _apply_h(Hf[t], poly_ref)
        if poly[:, 0].max() < 0 or poly[:, 1].max() < 0 or poly[:, 0].min() > Ww or poly[:, 1].min() > Hh:
            continue
        x0, y0 = int(max(0, poly[:, 0].min() - 4)), int(max(0, poly[:, 1].min() - 4))
        x1, y1 = int(min(Ww, poly[:, 0].max() + 4)), int(min(Hh, poly[:, 1].max() + 4))
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        crop = srgb_to_lin(f32(frames[t][y0:y1, x0:x1])[..., :3])
        pm = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(pm, [(poly - [x0, y0]).astype(np.int32)], 1)
        wv = np.ones_like(pm, np.float32) if mattes is None else np.asarray(mattes[t], np.float32)[y0:y1, x0:x1]
        sel = (pm > 0) & (wv > 0.5)
        if sel.sum() < 20:
            continue
        S, C, _ = shader.light(crop, np.maximum(wv, 1e-3), light_surface)
        S_list.append(float(S[sel].mean()))
        C_list.append(C[sel].mean(0))
        P97.append(float(np.percentile(S[sel], 97)))
    S_ref = float(np.median(S_list)) if S_list else 0.5
    C_ref = np.median(np.stack(C_list), 0) if C_list else np.ones(3, np.float32)
    white_cap = float(np.median(P97)) if (P97 and light_surface and lum_ref_surface > 0.25) else None
    rep["shading"] = {"typical_light": round(S_ref, 4), "light_range": [round(min(S_list) / S_ref, 3),
                      round(max(S_list) / S_ref, 3)] if S_list else None, "surface": "light" if light_surface else "dark",
                      "ink_capped_to_surface_white": bool(white_cap is not None and _lum(ink_lin) > white_cap)}

    # -- occlusion check (appearance) -------------------------------------------------------------------------
    occ_masks, occ = [None] * n, None
    if occlusion_check:
        occ = OcclusionCheck(ref_f, lay, place["box"], place["cap_tex"] * k_ref, (Hh, Ww), surf_full)
        occ_masks, ost = occ.run(lambda t: f32(frames[t]), Hf, zoom, n, ref, log)
        rep["occlusion_check"] = ost

    # -- per frame -----------------------------------------------------------------------------------------
    patches = []
    blur_px, occl_frames, hidden_frames, out_frames = [], 0, 0, 0
    marg = shade_margin(shader)
    for t in range(n):
        Hs, wts, blen = blur_set(Hf, t, quad_ref, P_mo, V_mo, motion_blur)
        blur_px.append(blen)
        r = text_alpha(tex, lay, Hs, wts, (Hh, Ww))
        if r is None:
            patches.append(None)
            out_frames += 1
            continue
        a, x0, y0 = r
        bh, bw = a.shape
        X0, Y0, X1, Y1 = max(0, x0 - marg), max(0, y0 - marg), min(Ww, x0 + bw + marg), min(Hh, y0 + bh + marg)
        region = f32(frames[t][Y0:Y1, X0:X1])
        vis = np.ones((Y1 - Y0, X1 - X0), np.float32) if mattes is None else \
            np.asarray(mattes[t][Y0:Y1, X0:X1], np.float32)
        if mattes is not None:  # semi-transparent matte values inside a surface (glass) don't fade the ink
            vis = np.clip((vis - 0.15) / 0.5, 0, 1)
        if occluders is not None:
            vis = vis * (1 - np.asarray(occluders[t][Y0:Y1, X0:X1], np.float32))
        if occ_masks[t] is not None:
            vis = vis * (1 - occ.in_frame(occ_masks[t], Hf[t], X0, Y0, X1 - X0, Y1 - Y0))
        rgb, a_eff = composite(region, a, x0 - X0, y0 - Y0, ink_lin, vis, shader, S_ref, C_ref, white_cap,
                               light_surface)
        full_cov = float(a.sum())
        if full_cov > 0:
            hidden = 1 - float(a_eff.sum()) / full_cov
            info[t]["text_hidden"] = round(hidden, 3)
            if hidden > 0.02:
                occl_frames += 1
            if hidden > 0.98:
                hidden_frames += 1
        patches.append({"y0": y0, "x0": x0, "rgb": rgb, "alpha": a_eff})
        if t % 40 == 0:
            log(f"frame {t}/{n}")

    # -- report ---------------------------------------------------------------------------------------------
    st = [i["status"] for i in info]
    cc = [i["cc"] for i in info if i["status"] in ("tracked", "reacquired")]
    centre = np.array([_apply_h(Hf[t], ref_pts.mean(0, keepdims=True))[0] for t in range(n)])
    jit = 0.0
    if n >= 5:
        acc2 = centre[2:] - 2 * centre[1:-1] + centre[:-2]
        jit = float(np.sqrt(np.median((acc2 ** 2).sum(1))))
        cr = np.array([_apply_h(rep["_H_raw"][t], ref_pts.mean(0, keepdims=True))[0] for t in range(n)])
        a2 = cr[2:] - 2 * cr[1:-1] + cr[:-2]
        rep["_accel_raw"] = float(np.sqrt(np.median((a2 ** 2).sum(1))))
    rep["tracking"] = {
        "method": "KLT + RANSAC prediction, band-pass ECC against the reference frame (a recent anchor while the "
                  "look drifts), SIFT re-acquisition, sub-pixel jitter smoothing",
        "anchors_used": len(set(i.get("anchor", ref) for i in info)),
        "status_counts": {k: st.count(k) for k in sorted(set(st))},
        "ecc_cc_mean": round(float(np.mean(cc)), 4) if cc else None,
        "ecc_cc_min": round(float(np.min(cc)), 4) if cc else None,
        "accel_median_px": round(jit, 3),
        "accel_median_px_before_smoothing": round(rep.pop("_accel_raw", 0.0), 3),
        "per_frame": [{"f": i, **info[i]} for i in range(n)],
    }
    rep["motion_blur_px"] = {"max": round(max(blur_px), 2), "frames_blurred": int(sum(b >= 1.0 for b in blur_px))}
    rep["frames_text_partly_hidden"] = occl_frames
    rep["frames_text_hidden"] = hidden_frames
    rep["frames_text_out_of_frame"] = out_frames
    t2d = [i.get("texture_2d") for i in info if i.get("texture_2d") is not None]
    if t2d and float(np.median(t2d)) < ONE_WAY:
        rep["review_reasons"].append("The surface's detail runs one way (grain, stripes or a thin strip): the text "
                                     "may slide along it; a matte of the whole object or a surface with more detail helps")
    bad = st.count("predicted") + st.count("held")
    if bad:
        rep["review_reasons"].append(f"{bad} frames could not be locked to the surface (motion predicted): check them")
    if cc and np.mean(cc) < 0.75:
        rep["review_reasons"].append("The surface changes a lot against the reference frame (low alignment score)")
    if lay.kind == "plane" and lrep.get("surface", "").startswith("Flat (facing"):
        rep["notes"].append("The surface outline isn't four-sided: the text faces the reference view")
    if rep["cap_height_px"]["min"] < 9:
        rep["review_reasons"].append(f"The text is only {rep['cap_height_px']['min']} px tall in some frames: "
                                     "check legibility (legal supers usually need more)")
    if lrep.get("shape_test", {}).get("kind") == "round_or_flat" and shape == "Auto":
        rep["notes"].append("Round or flat? The sides are straight but the base isn't curved: used flat")
    rep["needs_review"] = bool(rep["review_reasons"])
    rep["_H"] = Hf
    rep["_layout"] = (lay, place)
    return patches, rep


# ------------------------------------------------------------------------------------------------- ComfyUI
class AutoMatteTextOnSurface:
    """Put text on a moving surface so it looks printed: a price on a box, a claim on a label, a legal super on a
    wall or a table, a name on a shirt. The matte is the surface (white = where the text may go; anything in front of
    the surface is black in it, so the text goes behind). The text follows the surface's motion (tracked against one
    reference frame: no drift), wraps around round surfaces, lies in a flat surface's perspective, takes the
    surface's light and gets the clip's motion blur. Pixels outside the text are unchanged. Outputs the frames, the
    text's own matte (for compositing elsewhere) and a report with tracking quality per frame and review reasons."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "matte": ("MASK", {"tooltip": "White = the surface the text goes on (from Video Segmentation or SAM)."}),
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "The text; new lines start new lines."}),
            },
            "optional": {
                "placement": (PLACEMENTS, {"default": "Center", "tooltip": "Where on the surface; Emptiest Area finds the plainest spot."}),
                "size": (list(SIZES), {"default": "Auto", "tooltip": "Auto fits the text to the surface."}),
                "font": (list(FONTS), {"default": "Montserrat"}),
                "bold": ("BOOLEAN", {"default": False}),
                "colour": ("STRING", {"default": "", "tooltip": "Hex, rgb() or a colour name, as seen under the clip's typical light. Empty = contrasting ink."}),
                "font_file": ("STRING", {"default": "", "tooltip": "Path to a brand font (TTF/OTF) instead of a bundled one."}),
                "surface_shape": (["Auto", "Flat", "Round"], {"default": "Auto", "tooltip": "Round wraps the text around a bottle or can."}),
                "motion_blur": ("BOOLEAN", {"default": True}),
                "occlusion_check": ("BOOLEAN", {"default": False, "tooltip": "Also hide the text where something the matte misses covers the surface (by look and motion)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "text_matte", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    def run(self, images, matte, text, placement="Center", size="Auto", font="Montserrat", bold=False,
            colour="", font_file="", surface_shape="Auto", motion_blur=True, occlusion_check=False):
        import torch
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        if not (text or "").strip():
            return (images, torch.zeros((n, h, w)), json.dumps({"edit": "text_on_surface", "notes": ["No text given"],
                                                                 "needs_review": False}))
        m = matte.float()
        if m.shape[1:] != (h, w):
            m = torch.nn.functional.interpolate(m[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        if m.shape[0] < n:
            m = torch.cat([m, m[-1:].repeat(n - m.shape[0], 1, 1)], 0)
        frames = [images[i, ..., :3].cpu().numpy() for i in range(n)]
        mattes = [m[i].clamp(0, 1).cpu().numpy() for i in range(n)]
        patches, rep = text_on_surface(frames, text, mattes=mattes, placement=placement, size=size, font=font,
                                       bold=bold, font_file=(font_file or None), colour=colour, shape=surface_shape,
                                       motion_blur=motion_blur, occlusion_check=occlusion_check)
        res = images.clone()
        am = torch.zeros((n, h, w), dtype=torch.float32)
        for i, p in enumerate(patches):
            if p is None:
                continue
            y0, x0 = p["y0"], p["x0"]
            ph, pw = p["alpha"].shape
            res[i, y0:y0 + ph, x0:x0 + pw, :3] = torch.from_numpy(p["rgb"]).to(res.dtype)
            am[i, y0:y0 + ph, x0:x0 + pw] = torch.from_numpy(p["alpha"])
        for k in ("_H", "_H_raw", "_layout"):
            rep.pop(k, None)
        return (res, am, json.dumps(rep))


NODE_CLASS_MAPPINGS = {"AutoMatteTextOnSurface": AutoMatteTextOnSurface}
NODE_DISPLAY_NAME_MAPPINGS = {"AutoMatteTextOnSurface": "AutoMatte: Text on Surface"}
