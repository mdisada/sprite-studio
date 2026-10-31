"""Character definitions: reference-image prompt + animation prompts.

Design notes for `stormshield`:
  Role: ranged magic DPS. Lightning/AoE magic channeled through or bounced
  off the shield. Team passive: reduces ally cooldowns (clock/tempo runes on
  the shield). Unique passive: basic attacks can chain to extra enemies
  (forked/chain lightning motif).
"""

# Shared style block for still-image generation (Krea-2 Turbo).
PIXEL_STYLE = (
    "32-bit era video game pixel art character sprite, detailed high-color pixel "
    "shading with clean dithering, crisp dark outline, strong readable silhouette, "
    "premium SNES / Saturn arcade fighting-game sprite quality. "
)

# Shared framing/background block: flat green so we can chroma-key it out.
GREEN_STAGE = (
    "The character is perfectly centered, whole body visible with a clear margin on "
    "all sides, feet at the bottom, no cropping. Background is a perfectly flat, "
    "uniform, solid bright green chroma-key backdrop with no shadows, no floor line, "
    "no props, no text, no UI, single character only."
)

# Shared constraints for MiniMax H3 animation clips.
# Strategy (user's call): pixel-perfection is enforced on the REFERENCE
# (grid-conformed before upload) and re-imposed in post (k-centroid +
# palette snap), so the clip prompt only needs to keep the retro STYLE -
# demanding literal hard pixels per frame from a video model is fragile.
# Anti-3D / no-motion-blur stay: blur destroys the post-time grid snap.
ANIM_RULES = (
    "<Picture 1> defines the character's exact identity, outfit, proportions, "
    "colors and 32-bit pixel-art game-sprite style - keep that retro look in "
    "every frame. Fluid natural 2D animation; do not re-render in 3D, no "
    "photorealism, no motion blur, colors stay flat and graphic with clear "
    "dark outlines. Locked-off single static wide shot: the frame never moves "
    "- no pan, no zoom, no push-in, no reframing, no cuts, no handheld. The "
    "character stays perfectly centered at constant scale, full body always "
    "visible, feet on a fixed baseline. The background stays a perfectly flat "
    "solid bright green screen in every frame: no floor, no shadows, no "
    "environment, no props, no particles behind the character, no text. "
    "No speech, no dialogue, no mouth movement. The character's face, eyes, "
    "feet and body all point the same way - toward the right side of the "
    "frame - and the head never turns to the left. All motion plays at "
    "natural real-time speed with athletic, snappy timing - never in slow "
    "motion, never floaty. "
)

# --- painted (AFK-Arena-like) style templates ------------------------------
# Smooth hand-painted hero art instead of a pixel grid: what the video model
# produces most naturally, and the mobile-RPG presentation style.

PAINTED_STYLE = (
    "High-quality stylized 2D mobile-RPG hero character art: hand-painted "
    "with clean bold dark outlines, rich painterly cel shading, soft color "
    "gradients, vibrant saturated colors, slightly exaggerated heroic "
    "proportions with a strong readable silhouette, polished professional "
    "game art quality, smooth crisp edges, no pixelation. "
)

PAINTED_ANIM_RULES = (
    "<Picture 1> defines the character's exact identity, outfit, proportions, "
    "colors and hand-painted art style. Smooth 2D hand-painted game "
    "animation: clean bold outlines and painterly cel shading in every "
    "frame, fluid natural motion. Do not re-render in 3D, no photorealism, "
    "no pixelation. Locked-off single static wide shot: the frame never "
    "moves - no pan, no zoom, no push-in, no reframing, no cuts. The "
    "character stays perfectly centered at constant scale, full body always "
    "visible, feet on a fixed baseline. The background stays a perfectly "
    "flat solid bright green screen in every frame: no floor, no shadows, "
    "no environment, no props, no text. No speech, no mouth movement. "
    "The character's face, eyes, feet and body all point the same way - "
    "toward the right side of the frame - and the head never turns to the "
    "left. All motion plays at natural real-time speed with athletic, "
    "snappy timing - never in slow motion, never floaty. "
)


# --- project-level presets ---------------------------------------------------
# A project fixes the art direction once: style (pixel vs painted), pixel
# density for pixel projects, and the game type (default movement kit).

# Pixel-density presets by console "bit" era. sprite_h is chosen so the
# 640px final video divides by an exact integer (the ONLY way the pixel
# grid stays consistent): 640/20=32, 640/10=64, 640/5=128, 640/4=160.
# The style text goes in front of reference prompts; anim_rules replaces
# the generic pixel rules in clip prompts.
_ERA_RULES = lambda era: ANIM_RULES.replace("32-bit", era)

PIXEL_PRESETS = {
    "8": {
        "label": "8-bit (NES / Game Boy era)",
        "blurb": "tiny chunky sprites, ~8 flat colors, bold outlines",
        "sprite_h": 32, "colors": 6,
        "fps": 8,  "frames_loop": 6,  "frames_once": 10,
        "style": ("8-bit NES-era video game pixel art character sprite, chunky "
                  "large pixels, small simplified figure, very limited flat "
                  "color palette, no dithering, no anti-aliasing, hard black "
                  "outline, bold readable silhouette. "),
        "anim_rules": _ERA_RULES("8-bit"),
    },
    "16": {
        "label": "16-bit (SNES / Genesis era)",
        "blurb": "classic 64px sprites, 16 colors, selective dither",
        "sprite_h": 64, "colors": 16,
        "fps": 10, "frames_loop": 10, "frames_once": 16,
        "style": ("16-bit SNES-era video game pixel art character sprite, clean "
                  "pixel clusters, limited 16-color palette, sparing dithering, "
                  "crisp dark outline, strong readable silhouette. "),
        "anim_rules": _ERA_RULES("16-bit"),
    },
    "32": {
        "label": "32-bit (Saturn / arcade / Metal Slug era)",
        "blurb": "detailed 128px sprites, rich palette, premium arcade look",
        "sprite_h": 128, "colors": 32,
        "fps": 12, "frames_loop": 16, "frames_once": 24,
        "style": PIXEL_STYLE,
        "anim_rules": ANIM_RULES,
    },
    "64": {
        "label": "Hi-bit (modern indie: Owlboy / Eastward)",
        "blurb": "fine 160px sprites, painterly pixel shading, big palette",
        "sprite_h": 160, "colors": 64,
        "fps": 12, "frames_loop": 20, "frames_once": 30,
        "style": ("Modern high-bit indie pixel art character sprite in the "
                  "Owlboy / Eastward tradition: fine pixel detail, rich "
                  "palette, painterly pixel-cluster shading, crisp selective "
                  "outlines, premium contemporary pixel art quality. "),
        "anim_rules": _ERA_RULES("high-bit"),
    },
}

# Default movement kits by game type. Each entry: animation key, the
# ANIM_PRESETS prompt it scaffolds from (webui), canvas hint, and an
# optional pose the animation needs first (top-down directions need a
# turned reference pose before they can be animated).
GAME_KITS = {
    "platformer": {
        "label": "Platformer (Mario / Megaman)",
        "blurb": "side-view, one facing, mirrored in-engine",
        "anims": [
            {"key": "idle", "preset": "idle"},
            {"key": "walk", "preset": "walk"},
            {"key": "run", "preset": "run"},
            {"key": "jump", "preset": "jump", "canvas": "portrait"},
            {"key": "fall", "preset": "fall", "canvas": "portrait"},
            {"key": "attack", "preset": "attack"},
            {"key": "hit", "preset": "hit"},
            {"key": "death", "preset": "death"},
        ],
    },
    "side_arena": {
        "label": "Side-view arena (AFK Arena style)",
        "blurb": "3/4 side view, idle-battler presentation",
        "anims": [
            {"key": "idle", "preset": "idle"},
            {"key": "walk", "preset": "walk"},
            {"key": "attack", "preset": "attack"},
            {"key": "skill_cast", "preset": "skill"},
            {"key": "hit", "preset": "hit"},
            {"key": "death", "preset": "death"},
            {"key": "victory", "preset": "victory"},
        ],
    },
    "topdown": {
        "label": "Top-down RPG (Pokemon / Zelda)",
        "blurb": "4 directions: down, up, side (left = mirrored right)",
        "anims": [
            {"key": "idle_down", "preset": "idle_down", "needs_view": "facing_down"},
            {"key": "walk_down", "preset": "walk_down", "needs_view": "facing_down"},
            {"key": "idle_up", "preset": "idle_up", "needs_view": "facing_up"},
            {"key": "walk_up", "preset": "walk_up", "needs_view": "facing_up"},
            {"key": "idle_side", "preset": "idle_side", "needs_view": "facing_side"},
            {"key": "walk_side", "preset": "walk_side", "needs_view": "facing_side"},
            {"key": "hit", "preset": "hit"},
            {"key": "death", "preset": "death"},
        ],
        "views": {
            "facing_down": "Turn the character to face straight toward the viewer (front view), arms relaxed at the sides, symmetrical stance.",
            "facing_up": "Turn the character to face directly away from the viewer (back view), showing the back of the head and outfit, symmetrical stance.",
            "facing_side": "Turn the character into a strict side profile facing the right side of the frame.",
        },
    },
    "isometric": {
        "label": "Isometric (Diablo / Tactics)",
        "blurb": "SE + NE directions scaffolded; mirror for SW/NW in-engine",
        "anims": [
            {"key": "idle_se", "preset": "idle", "needs_view": "facing_se"},
            {"key": "walk_se", "preset": "walk", "needs_view": "facing_se"},
            {"key": "idle_ne", "preset": "idle", "needs_view": "facing_ne"},
            {"key": "walk_ne", "preset": "walk", "needs_view": "facing_ne"},
            {"key": "attack_se", "preset": "attack", "needs_view": "facing_se"},
            {"key": "hit", "preset": "hit"},
            {"key": "death", "preset": "death"},
        ],
        "views": {
            "facing_se": "Turn the character to a three-quarter front view facing down-right, as seen from an elevated isometric camera angle.",
            "facing_ne": "Turn the character to a three-quarter back view facing up-right, showing mostly their back, as seen from an elevated isometric camera angle.",
        },
    },
}


# Pixel-art flavor presets for reference variation batches in pixel
# projects (the painted STYLE_PRESETS below would break the pixel look).
PIXEL_STYLE_PRESETS = {
    "clean_cel": ("Clean cel-shaded pixel art with flat color planes, minimal "
                  "dithering, bold dark outlines, bright saturated palette. "),
    "arcade_dither": ("Gritty 90s arcade pixel art with heavy skilled "
                      "dithering, dramatic contrast, deep shadows, Metal Slug "
                      "/ Neo Geo energy. "),
    "painterly_hibit": ("Soft painterly hi-bit pixel art with subtle cluster "
                        "shading and rich color ramps, Owlboy-like warmth. "),
    "cartoon_bold": ("Cartoony pixel art with extra-thick outlines, chunky "
                     "shapes, exaggerated proportions, playful palette. "),
    "muted_retro": ("Muted retro-console pixel art with a restrained earthy "
                    "palette, low contrast, cozy 16-bit RPG mood. "),
    "neon_dark": ("Dark neon pixel art with moody shadows and vivid glowing "
                  "accent colors, cyber-fantasy contrast. "),
}

# Art-direction presets for reference variation batches (same character,
# different style axis - seeds alone barely change the look).
STYLE_PRESETS = {
    "chibi": ("Cute chibi mobile-game hero art: super-deformed proportions "
              "with a big head, bold clean outlines, bright cel shading, "
              "polished gacha-game quality. "),
    "afk_painterly": ("Stylized western-fantasy mobile-RPG hero art in the "
                      "AFK Arena tradition: rich hand-painted rendering, soft "
                      "volumetric shading, elegant exaggerated shapes, "
                      "slightly stylized proportions - NOT chibi, NOT anime. "),
    "anime_rpg": ("High-quality anime RPG character art: realistic human "
                  "proportions, crisp clean anime cel shading, detailed "
                  "costume rendering, expressive detailed eyes - NOT chibi. "),
    "inked_comic": ("Bold graphic-novel game art like modern action "
                    "roguelikes: thick expressive ink outlines, dramatic rim "
                    "lighting, saturated colors with painted accents. "),
    "storybook": ("Soft storybook fantasy painting: gentle watercolor-like "
                  "brushwork, warm diffused light, rounded friendly shapes, "
                  "delicate fine linework. "),
    "jrpg_tactics": ("Classic JRPG tactical-RPG character art: painterly "
                     "rendering with visible brush texture, rich muted "
                     "palette, ornate detailed armor, grounded fantasy "
                     "illustration quality - NOT chibi. "),
}


# --- effect (VFX) sprite templates -----------------------------------------
# Effects are generated as their own sprites and overlaid on characters
# in-engine, so they can fill their whole frame instead of being cramped
# into a character clip (where they get cut off at the edges).

VFX_PIXEL = (
    "32-bit era video game pixel art visual effect sprite, luminous glowing "
    "energy drawn in crisp pixel clusters with clean dithering, high color "
    "count, premium SNES arcade quality. "
)

VFX_STAGE = (
    "The effect is centered with a small margin inside the frame and fully "
    "visible, no cropping. Background is a perfectly flat, uniform, solid "
    "bright green chroma-key backdrop with no shadows, no floor, no objects, "
    "no characters, no text, single effect only."
)

VFX_RULES = (
    "<Picture 1> defines the exact look and pixel-art style of this visual "
    "effect. 2D pixel-art game VFX animation: crisp handcrafted 32-bit pixel "
    "art with hard pixel edges and a limited retro palette in every frame. Do "
    "not smooth, do not re-render in 3D; no motion blur, no anti-aliasing, no "
    "photorealism. Locked-off single static shot: the frame never moves - no "
    "pan, no zoom, no reframing, no cuts. The effect stays centered and fully "
    "inside the frame on a perfectly flat solid bright green screen in every "
    "frame: no floor, no shadows, no characters, no objects, no text. No "
    "speech, no sound-driven visuals. "
)

# Appended to character skill/attack prompts: big effects live in separate
# VFX sprites, so the character clip must keep its effects body-contained.
CONTAINED_FX = (
    "All magic effects stay small and close to the character's body and "
    "shield; no bolt, beam or ring travels away from the character and "
    "nothing reaches the frame edges (projectiles and big effects are "
    "separate overlay sprites). "
)

CHARACTERS = {
    "stormshield": {
        "display_name": "Stormshield Arcanist",
        "ref_prompt": (
            PIXEL_STYLE
            + "Full body, side-facing three-quarter view, standing combat-idle pose. "
            "A storm-mage skirmisher: a slender battle-caster in deep indigo-blue "
            "layered robes with silver-and-gold trim over light leather armor, short "
            "storm-grey cape, high collar. On the left arm a large ornate round "
            "shield with a polished mirror-bright steel face etched with glowing "
            "cyan lightning runes and a ring of small clock-like tempo runes around "
            "the rim, held angled forward like a focusing lens. The right hand is "
            "raised, crackling with cyan-white electricity, and thin forked arcs of "
            "chain lightning bounce off the shield face. Glowing cyan eyes, calm "
            "focused expression. "
            + GREEN_STAGE
        ),
        "animations": {
            "idle": {
                "prompt": ANIM_RULES + (
                    "Idle animation loop: the storm-mage breathes subtly, the cape and "
                    "robe hems sway gently, and small arcs of cyan lightning flicker "
                    "now and then across the rim of the round shield. The movement is "
                    "small and calm, and the pose at the end matches the pose at the "
                    "start so the loop is seamless."
                ),
                "seconds": 5.2,
            },
            "cast": {
                "prompt": ANIM_RULES + (
                    "Attack animation: the storm-mage swings the round mirrored shield "
                    "up and forward, channels a bright surge of crackling cyan "
                    "electricity from the raised right hand into the shield face, and "
                    "the shield fires the bolt forward toward the right side of the "
                    "frame as forked chain lightning that splits into multiple arcs. "
                    "Bright flash at the moment of release, sparks scatter, slight "
                    "recoil step back, then the mage settles back into the guard pose."
                ),
                "seconds": 5.2,
            },
            "walk": {
                "prompt": ANIM_RULES + (
                    "Walk cycle animation: the storm-mage walks in place with a steady "
                    "rhythmic gait facing the right side of the frame, shield held "
                    "forward, cape bobbing with each step, feet clearly stepping. The "
                    "cycle is even and loopable and the character does not travel "
                    "across the frame, staying centered."
                ),
                "seconds": 5.2,
            },
        },
    },
}
