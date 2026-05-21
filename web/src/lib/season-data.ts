export type SeasonKey =
  | "spring_warm" | "spring_light" | "spring_bright"
  | "summer_cool" | "summer_light" | "summer_soft"
  | "autumn_warm" | "autumn_soft" | "autumn_deep"
  | "winter_cool" | "winter_deep" | "winter_bright";

export interface ColorSwatch {
  name: string;
  hex: string;
}

export interface ClothingCategory {
  wear: string;
  avoid?: string;
}

export interface SeasonResult {
  displayName: string;
  season: "spring" | "summer" | "autumn" | "winter";
  description: string;
  bestColors: ColorSwatch[];
  avoidColors: ColorSwatch[];
  clothes: {
    tops: ClothingCategory;
    bottoms: ClothingCategory;
    bagsShoes: ClothingCategory;
    metals: ClothingCategory;
    pairingTips: ClothingCategory;
  };
  makeup: {
    lips: ClothingCategory;
    eyes: ClothingCategory;
    blush: ClothingCategory;
  };
  accentColor: string;
  bgColor: string;
  textColor: string;
}

export const SEASON_DATA: Record<SeasonKey, SeasonResult> = {
  winter_cool: {
    displayName: "Cool Winter",
    season: "winter",
    description:
      "Defined by cool undertones, high contrast, and a naturally bold, striking appearance. Your best colors are pure, icy, or deeply saturated: true white, cobalt, navy, emerald, true red, fuchsia, hot pink, burgundy, charcoal, and black. Silver jewelry will always flatter you more than gold. Avoid anything warm, earthy, or muted — beige, camel, rust, terracotta, olive, peach, orange, ivory, and warm yellows will wash out your natural vibrancy. If it feels like autumn or warmth, skip it; if it feels icy, bold, or crisp, it's made for you.",
    bestColors: [
      { name: "Icy blue", hex: "#B8D8F0" },
      { name: "Royal blue", hex: "#3A60D8" },
      { name: "True red", hex: "#C80000" },
      { name: "Hot pink", hex: "#E8509A" },
      { name: "Burgundy", hex: "#7A001C" },
      { name: "Charcoal", hex: "#3A4858" },
      { name: "Emerald", hex: "#2A9C6A" },
      { name: "Ice pink", hex: "#F0C0D8" },
      { name: "Cobalt", hex: "#0040A0" },
      { name: "Fuchsia", hex: "#C030A0" },
      { name: "Navy", hex: "#08205A" },
      { name: "Ice lavender", hex: "#D8D0F0" },
    ],
    avoidColors: [
      { name: "Warm beige", hex: "#C4A882" },
      { name: "Camel", hex: "#C09060" },
      { name: "Orange", hex: "#E07030" },
      { name: "Yellow", hex: "#E8C030" },
      { name: "Rust", hex: "#B03010" },
      { name: "Warm brown", hex: "#8B5030" },
      { name: "Olive", hex: "#787830" },
      { name: "Peach", hex: "#F5C090" },
      { name: "Terracotta", hex: "#C07050" },
      { name: "Gold", hex: "#D4A020" },
      { name: "Tomato red", hex: "#E04030" },
      { name: "Warm mauve", hex: "#C08888" },
    ],
    clothes: {
      tops: {
        wear: "True red, cobalt blue, fuchsia, hot pink, icy lavender, emerald, crisp white, black",
        avoid: "Peach, camel, rust — these dull your complexion",
      },
      bottoms: {
        wear: "Black, charcoal, navy, burgundy, cobalt — cool neutrals and jewel tones",
        avoid: "Warm browns, olive, beige",
      },
      bagsShoes: {
        wear: "Black, white, navy, deep burgundy — anchors any outfit",
        avoid: "Tan, camel, warm brown leather",
      },
      metals: {
        wear: "Silver, white gold, platinum, diamonds, sapphires, rubies, emeralds, amethysts, and pearls — these complement your cool undertones.",
        avoid: "Yellow gold, rose gold, and warm/yellow-tinted stones — they add warmth that clashes with your coloring.",
      },
      pairingTips: {
        wear: "High contrast works in your favor — bold top + black/navy bottoms. All-black or black & white are signature Cool Winter looks.",
      },
    },
    makeup: {
      lips: {
        wear: "Blue-based reds, true reds, berry, fuchsia, hot pink, deep burgundy. Nude lips must lean cool and rosy.",
        avoid: "Warm, peachy, or brown-toned nudes.",
      },
      eyes: {
        wear: "Icy pinks, silver, charcoal, navy, deep plums. Black eyeliner and mascara enhance your natural contrast.",
        avoid: "Warm browns, bronzes, and gold eyeshadow.",
      },
      blush: {
        wear: "Cool or neutral-pink blushes: rose, raspberry, soft fuchsia. Foundation must always be cool or neutral.",
        avoid: "Peachy or coral blush — never warm or yellow-based.",
      },
    },
    accentColor: "#3A60D8",
    bgColor: "#08080F",
    textColor: "#FFFFFF",
  },

  winter_deep: {
    displayName: "Deep Winter",
    season: "winter",
    description:
      "Rich, deep, and intensely cool — your palette thrives on high contrast and saturated depth. You carry the deepest, most dramatic colors of the Winter family with ease.",
    bestColors: [
      { name: "True black", hex: "#080808" },
      { name: "Deep navy", hex: "#06102A" },
      { name: "Burgundy", hex: "#6A0018" },
      { name: "Deep plum", hex: "#4A0848" },
      { name: "Forest green", hex: "#0E3818" },
      { name: "Charcoal", hex: "#303838" },
      { name: "Cobalt", hex: "#1030A0" },
      { name: "Crimson", hex: "#9C0020" },
      { name: "Deep teal", hex: "#084848" },
      { name: "Midnight blue", hex: "#080828" },
      { name: "Espresso", hex: "#381818" },
      { name: "True white", hex: "#F8F8F8" },
    ],
    avoidColors: [
      { name: "Pale peach", hex: "#F0C8A0" },
      { name: "Light camel", hex: "#D0A870" },
      { name: "Tan", hex: "#C09060" },
      { name: "Coral", hex: "#E87060" },
      { name: "Warm ivory", hex: "#F0E0C0" },
      { name: "Dusty pink", hex: "#D0A0A0" },
      { name: "Terracotta", hex: "#B86848" },
      { name: "Golden brown", hex: "#A07020" },
      { name: "Mustard", hex: "#C89020" },
      { name: "Warm rust", hex: "#B04820" },
      { name: "Blush", hex: "#E8B8B0" },
      { name: "Warm gray", hex: "#A09080" },
    ],
    clothes: {
      tops: {
        wear: "Black, deep burgundy, midnight navy, forest green, crimson, deep plum, true white for contrast",
        avoid: "Pastels, warm corals, light peach, camel — these create an imbalance with your depth",
      },
      bottoms: {
        wear: "Black, deep charcoal, dark navy, espresso brown, midnight plum",
        avoid: "Khaki, tan, warm brown, medium gray — too light for your depth",
      },
      bagsShoes: {
        wear: "Black leather, deep burgundy, dark navy, midnight green",
        avoid: "Tan, camel, nude, or any warm-toned leather",
      },
      metals: {
        wear: "Silver, gunmetal, dark steel, black metals, platinum, white gold",
        avoid: "Yellow gold, rose gold, brass, copper — warm metals fight your depth",
      },
      pairingTips: {
        wear: "All-black looks are your signature. Deep jewel tones create stunning monochromatic outfits. Use true white only as an accent, not as a main piece.",
      },
    },
    makeup: {
      lips: {
        wear: "Deep berry, wine, dark plum, cool raspberry, true red",
        avoid: "Warm nudes, corals, warm pinks — they look muddy against your skin",
      },
      eyes: {
        wear: "Espresso, deep charcoal, plum, midnight navy, black — heavy liner suits you beautifully",
        avoid: "Warm bronze, copper, peach, warm brown shadows",
      },
      blush: {
        wear: "Deep rose, cool berry, raspberry — keep it deep enough to show on your skin",
        avoid: "Peachy coral or warm pink blush — they look washed out against your depth",
      },
    },
    accentColor: "#6A0018",
    bgColor: "#060608",
    textColor: "#FFFFFF",
  },

  winter_bright: {
    displayName: "Bright Winter",
    season: "winter",
    description:
      "Electric, vivid, and high-contrast — you carry the brightest and most saturated colors with effortless impact. Where others would look overpowered, you're electrified.",
    bestColors: [
      { name: "True white", hex: "#FFFFFF" },
      { name: "True black", hex: "#080808" },
      { name: "Electric blue", hex: "#0050E8" },
      { name: "Hot pink", hex: "#F01890" },
      { name: "Bright red", hex: "#E00020" },
      { name: "Vivid emerald", hex: "#00A858" },
      { name: "Bright cobalt", hex: "#0038C8" },
      { name: "Vivid fuchsia", hex: "#D000B0" },
      { name: "Bright purple", hex: "#6000D0" },
      { name: "Lemon yellow", hex: "#E8E000" },
      { name: "Electric teal", hex: "#00B8B8" },
      { name: "Icy pink", hex: "#F8C0D8" },
    ],
    avoidColors: [
      { name: "Dusty rose", hex: "#C89898" },
      { name: "Muted sage", hex: "#909870" },
      { name: "Warm beige", hex: "#C8A880" },
      { name: "Soft terracotta", hex: "#C07858" },
      { name: "Muted lavender", hex: "#B0A0C0" },
      { name: "Khaki", hex: "#B0A070" },
      { name: "Camel", hex: "#C09060" },
      { name: "Blush", hex: "#E8B0B0" },
      { name: "Warm tan", hex: "#C0A080" },
      { name: "Muted olive", hex: "#909050" },
      { name: "Peach", hex: "#F0B890" },
      { name: "Dusty mauve", hex: "#C89898" },
    ],
    clothes: {
      tops: {
        wear: "Vivid true colors only — bright red, electric blue, hot pink, vivid emerald, pure white, true black",
        avoid: "Anything muted, dusty, or pastel — you need full saturation",
      },
      bottoms: {
        wear: "True black, bright white, vivid cobalt, electric emerald — keep it bold",
        avoid: "Gray, beige, khaki, dusty pastels, or muted colors",
      },
      bagsShoes: {
        wear: "True black, bright white, vivid red or electric cobalt as statement pieces",
        avoid: "Nude, tan, warm tones, or muted colors",
      },
      metals: {
        wear: "Bright silver, platinum, white gold — the shinier the better",
        avoid: "Gold, rose gold, bronze, copper",
      },
      pairingTips: {
        wear: "Monochromatic brights are your power move. White + one vivid color creates your most flattering look. Never dilute with muted tones.",
      },
    },
    makeup: {
      lips: {
        wear: "Vivid true red, bright fuchsia, electric berry, icy cool pink",
        avoid: "Warm nudes, corals, dusty mauves — these dull your natural vividness",
      },
      eyes: {
        wear: "Clean silver, pure white, vivid blues and purples, deep black liner",
        avoid: "Warm bronzes, peach, or any muted colors",
      },
      blush: {
        wear: "Bright cool pink, vivid rose — keep color saturated",
        avoid: "Peachy, warm, or dusty blush tones",
      },
    },
    accentColor: "#E00020",
    bgColor: "#080810",
    textColor: "#FFFFFF",
  },

  summer_cool: {
    displayName: "Cool Summer",
    season: "summer",
    description:
      "Soft, cool, and naturally elegant — your coloring thrives in muted, cool hues. Dusty roses, powdery blues, and smoky lavenders are your natural habitat. You look most beautiful in the quiet sophistication of understated color.",
    bestColors: [
      { name: "Dusty rose", hex: "#C89898" },
      { name: "Lavender", hex: "#B8A8D8" },
      { name: "Powder blue", hex: "#90B8D8" },
      { name: "Soft gray", hex: "#B0B8C8" },
      { name: "Mauve", hex: "#B89098" },
      { name: "Blush pink", hex: "#E8B8C0" },
      { name: "Soft teal", hex: "#78A8B0" },
      { name: "Periwinkle", hex: "#8898D0" },
      { name: "Dusty blue", hex: "#7898B8" },
      { name: "Muted berry", hex: "#9878A0" },
      { name: "Soft sage", hex: "#98B098" },
      { name: "Cool white", hex: "#F0F0F8" },
    ],
    avoidColors: [
      { name: "Warm orange", hex: "#E07030" },
      { name: "Mustard", hex: "#C89020" },
      { name: "Olive green", hex: "#787830" },
      { name: "Rust", hex: "#B04020" },
      { name: "Golden yellow", hex: "#D0A020" },
      { name: "Warm brown", hex: "#8B5030" },
      { name: "Terracotta", hex: "#C07050" },
      { name: "Camel", hex: "#C09060" },
      { name: "Warm red", hex: "#C03020" },
      { name: "Bright neon", hex: "#80F000" },
      { name: "Deep black", hex: "#000000" },
      { name: "Hot orange", hex: "#F06000" },
    ],
    clothes: {
      tops: { wear: "Dusty rose, soft lavender, powder blue, mauve, soft blush, periwinkle", avoid: "Warm orange, mustard, rust — these create a harsh contrast with your softness" },
      bottoms: { wear: "Soft gray, dusty blue, muted mauve, cool navy (not bright)", avoid: "Warm browns, camel, olive, khaki" },
      bagsShoes: { wear: "Soft taupe, dusty rose, cool gray, nude with pink undertones", avoid: "Warm tan, camel, golden brown" },
      metals: { wear: "Silver, rose gold (only the cooler shades), white gold, pewter", avoid: "Yellow gold, bronze, copper, brass" },
      pairingTips: { wear: "Tone-on-tone soft looks are your strongest. Mix dusty rose with mauve, or lavender with soft gray. Avoid sharp contrasts." },
    },
    makeup: {
      lips: { wear: "Cool pink, dusty rose, berry, soft mauve, sheer cool red", avoid: "Warm nudes, coral, warm orange-red" },
      eyes: { wear: "Taupe, muted lavender, soft gray, dusty mauve, cool brown", avoid: "Warm gold, copper, bronze, orange-based shadows" },
      blush: { wear: "Soft cool pink, muted rose, dusty berry — very lightly applied", avoid: "Warm peach, coral, warm apricot" },
    },
    accentColor: "#8898D0",
    bgColor: "#F8F8FC",
    textColor: "#1A1A2E",
  },

  summer_light: {
    displayName: "Light Summer",
    season: "summer",
    description: "Delicate, airy, and softly cool — you are the most ethereal of the Summers. Your light coloring needs equally light, soft, and cool tones that don't overpower your natural luminosity.",
    bestColors: [
      { name: "Ice pink", hex: "#F0D0D8" }, { name: "Soft lavender", hex: "#D0C8E8" },
      { name: "Baby blue", hex: "#C0D8F0" }, { name: "Mist gray", hex: "#D0D8E0" },
      { name: "Pale rose", hex: "#F0C8C8" }, { name: "Soft mint", hex: "#C0E0D8" },
      { name: "Periwinkle", hex: "#B8C0E0" }, { name: "Powder pink", hex: "#F0D0D8" },
      { name: "Light mauve", hex: "#D8B8C0" }, { name: "Cool ivory", hex: "#F8F0F0" },
      { name: "Sky blue", hex: "#B8D0E8" }, { name: "Soft white", hex: "#F8F8FA" },
    ],
    avoidColors: [
      { name: "Black", hex: "#000000" }, { name: "Deep navy", hex: "#081030" },
      { name: "Vivid orange", hex: "#E06020" }, { name: "Mustard", hex: "#C89020" },
      { name: "Dark brown", hex: "#402010" }, { name: "Warm rust", hex: "#B04020" },
      { name: "Olive", hex: "#707030" }, { name: "Camel", hex: "#C09060" },
      { name: "Bright neon", hex: "#80F000" }, { name: "Deep red", hex: "#800010" },
      { name: "Warm coral", hex: "#E07060" }, { name: "Golden yellow", hex: "#D0A020" },
    ],
    clothes: {
      tops: { wear: "Ice pink, soft lavender, baby blue, mist white, powder rose, very soft mint", avoid: "Black, deep navy, vivid colors, warm tones — they all overpower you" },
      bottoms: { wear: "Mist gray, soft pale blue, ivory white, light lavender", avoid: "Dark, heavy, or warm-toned colors" },
      bagsShoes: { wear: "Soft nude, pale blush, light gray, cool white", avoid: "Dark brown, black, warm camel" },
      metals: { wear: "Light silver, rose gold (very cool), delicate pearl, moonstone", avoid: "Heavy yellow gold, dark bronze, copper" },
      pairingTips: { wear: "Monochromatic soft looks are stunning on you. Layer different light cool tones together. Avoid strong contrast — it overwhelms your palette." },
    },
    makeup: {
      lips: { wear: "Sheer cool pink, soft rose, barely-there berry", avoid: "Dark, vivid, or warm-toned lips — always go light" },
      eyes: { wear: "Soft matte lavender, light gray, pale pink, barely-there taupe", avoid: "Heavy black liner, warm bronzes, orange-tinted shadows" },
      blush: { wear: "Very soft cool pink, sheer rose — barely visible is perfect", avoid: "Peach, warm coral, any warm blush" },
    },
    accentColor: "#B8C0E0",
    bgColor: "#F8F8FC",
    textColor: "#1A1A2E",
  },

  summer_soft: {
    displayName: "Soft Summer",
    season: "summer",
    description: "Muted, romantic, and perfectly balanced between warm and cool — Soft Summer is the most harmonious of all the seasons. Your natural coloring is gently blended, and your palette should mirror that soft, sophisticated harmony.",
    bestColors: [
      { name: "Dusty mauve", hex: "#B89090" }, { name: "Soft sage", hex: "#90A890" },
      { name: "Muted teal", hex: "#709090" }, { name: "Rose gray", hex: "#C0A8A8" },
      { name: "Lavender mist", hex: "#B0A8C8" }, { name: "Dusty blue", hex: "#8098A8" },
      { name: "Soft plum", hex: "#987088" }, { name: "Warm gray", hex: "#A8A0A0" },
      { name: "Muted coral", hex: "#C89888" }, { name: "Sage green", hex: "#88A080" },
      { name: "Rose beige", hex: "#C8B0A8" }, { name: "Soft white", hex: "#F5F0F0" },
    ],
    avoidColors: [
      { name: "True black", hex: "#000000" }, { name: "Vivid neon", hex: "#80F000" },
      { name: "Bright orange", hex: "#E05020" }, { name: "Pure white", hex: "#FFFFFF" },
      { name: "Mustard", hex: "#C89020" }, { name: "Electric blue", hex: "#0050E8" },
      { name: "Hot pink", hex: "#E81890" }, { name: "Lime green", hex: "#A0E020" },
      { name: "Vivid red", hex: "#E00020" }, { name: "Golden yellow", hex: "#D0A020" },
      { name: "Bright coral", hex: "#E06050" }, { name: "Stark contrast", hex: "#F0F0F0" },
    ],
    clothes: {
      tops: { wear: "Dusty mauve, soft sage, muted teal, rose beige, lavender mist, warm gray", avoid: "Anything vivid, stark, or high-contrast — your beauty is in the blend" },
      bottoms: { wear: "Muted gray-brown, dusty sage, soft rose-gray, quiet plum", avoid: "Black, white, very bright or very dark colors" },
      bagsShoes: { wear: "Rose beige, dusty taupe, muted sage leather, warm nude", avoid: "Black, stark white, vivid colored accessories" },
      metals: { wear: "Soft rose gold, matte silver, pewter, antique silver", avoid: "Shiny yellow gold, bright chrome silver, copper" },
      pairingTips: { wear: "Stay in your muted world — mix soft sage with dusty mauve, or rose gray with lavender mist. The more harmoniously blended, the better you look." },
    },
    makeup: {
      lips: { wear: "Muted rose, dusty pink, soft berry, quiet mauve", avoid: "Vivid reds, bright fuchsia, very dark plums, warm nudes" },
      eyes: { wear: "Soft taupe, dusty mauve, muted plum, gentle gray, cool rose-brown", avoid: "Stark black liner, vivid color, warm orange-brown, bronze" },
      blush: { wear: "Muted rose, soft dusty pink — buildable but never vivid", avoid: "Warm coral, peachy blush, vivid pink" },
    },
    accentColor: "#B89090",
    bgColor: "#F8F5F5",
    textColor: "#1A1A2E",
  },

  spring_warm: {
    displayName: "Warm Spring",
    season: "spring",
    description: "Golden, warm, and luminously alive — Warm Spring carries the sunniest, richest warm tones in the Spring family. Your palette is built on golden warmth, peachy brightness, and natural vitality.",
    bestColors: [
      { name: "Warm peach", hex: "#F0A878" }, { name: "Golden yellow", hex: "#E8C040" },
      { name: "Coral", hex: "#E87850" }, { name: "Warm turquoise", hex: "#40C8B0" },
      { name: "Ivory", hex: "#F8F0D8" }, { name: "Warm green", hex: "#70B048" },
      { name: "Salmon", hex: "#E88870" }, { name: "Amber", hex: "#D09030" },
      { name: "Warm tan", hex: "#C8A070" }, { name: "Bright coral", hex: "#E86040" },
      { name: "Light gold", hex: "#E8C878" }, { name: "Warm white", hex: "#F8F4E8" },
    ],
    avoidColors: [
      { name: "Cool gray", hex: "#9098A8" }, { name: "Icy blue", hex: "#B8D0E8" },
      { name: "Stark black", hex: "#000000" }, { name: "Cool purple", hex: "#8870C0" },
      { name: "Navy", hex: "#08205A" }, { name: "Burgundy", hex: "#700020" },
      { name: "Dark plum", hex: "#500848" }, { name: "Cool pink", hex: "#E090B0" },
      { name: "Charcoal", hex: "#303838" }, { name: "Silver", hex: "#B0B8C0" },
      { name: "Cool white", hex: "#F0F0F8" }, { name: "Deep teal", hex: "#085858" },
    ],
    clothes: {
      tops: { wear: "Warm peach, coral, golden yellow, ivory, warm green, salmon, light gold", avoid: "Cool gray, icy blue, stark black, cold purple — these drain your warmth" },
      bottoms: { wear: "Warm tan, light camel, ivory, warm beige, peachy nude", avoid: "Navy, charcoal, very dark or very cool colors" },
      bagsShoes: { wear: "Tan leather, warm camel, gold hardware, natural materials", avoid: "Black, stark silver, cold dark tones" },
      metals: { wear: "Yellow gold, rose gold, bronze, copper, warm brass", avoid: "Silver, platinum, white gold, chrome" },
      pairingTips: { wear: "Your power look is coral + ivory, or warm peach + gold. Keep everything in the warm, glowing spectrum and you'll radiate." },
    },
    makeup: {
      lips: { wear: "Coral, warm salmon, peachy pink, warm rose, true red with warm undertones", avoid: "Cool blue-based reds, cold berry, icy pink" },
      eyes: { wear: "Warm bronze, copper, golden brown, peach, warm champagne", avoid: "Cool gray, icy silver, cold-toned shadows" },
      blush: { wear: "Warm peach, apricot, coral — your most natural and flattering choice", avoid: "Cool pink, cool berry, muted mauve" },
    },
    accentColor: "#E87850",
    bgColor: "#FFF8F0",
    textColor: "#3A1A05",
  },

  spring_light: {
    displayName: "Light Spring",
    season: "spring",
    description: "Delicate, light, and warmly luminous — Light Spring has the softest and most gentle quality of all the Springs. Your natural coloring is light and clear, and your palette reflects that airy, sunny warmth.",
    bestColors: [
      { name: "Light peach", hex: "#F8D0A8" }, { name: "Pale gold", hex: "#F0D890" },
      { name: "Soft coral", hex: "#F0A888" }, { name: "Light aqua", hex: "#A8E0D8" },
      { name: "Warm ivory", hex: "#F8F0D8" }, { name: "Butter yellow", hex: "#F8E8A0" },
      { name: "Blush pink", hex: "#F8C8B0" }, { name: "Soft mint", hex: "#B8E8D0" },
      { name: "Light camel", hex: "#E0C090" }, { name: "Warm white", hex: "#F8F4E8" },
      { name: "Pale coral", hex: "#F8C8A8" }, { name: "Soft apricot", hex: "#F8D0B0" },
    ],
    avoidColors: [
      { name: "Black", hex: "#000000" }, { name: "Dark navy", hex: "#081030" },
      { name: "Deep burgundy", hex: "#600018" }, { name: "Bright neon", hex: "#80F000" },
      { name: "Cool gray", hex: "#9098A8" }, { name: "Dark olive", hex: "#505018" },
      { name: "Bright purple", hex: "#7020D0" }, { name: "Cool white", hex: "#F0F0F8" },
      { name: "Dark brown", hex: "#401808" }, { name: "Stark contrast", hex: "#000000" },
      { name: "Electric blue", hex: "#0050E8" }, { name: "Dark plum", hex: "#500848" },
    ],
    clothes: {
      tops: { wear: "Light peach, pale gold, soft coral, butter yellow, warm ivory, blush pink", avoid: "Black and very dark colors — they overpower your lightness" },
      bottoms: { wear: "Warm ivory, light camel, pale gold, soft warm beige", avoid: "Dark, cool, or very saturated colors" },
      bagsShoes: { wear: "Light tan, nude blush, warm white, pale gold accents", avoid: "Dark leather, stark black, cool tones" },
      metals: { wear: "Delicate yellow gold, rose gold, light bronze, pearl", avoid: "Heavy silver, dark metals, bold hardware" },
      pairingTips: { wear: "Light coral + warm ivory is your signature. Keep your whole look light and airy — never heavy or contrasting." },
    },
    makeup: {
      lips: { wear: "Sheer warm pink, light peach, pale coral, warm nude", avoid: "Dark, vivid, or cool-toned lips — always light" },
      eyes: { wear: "Light warm champagne, soft peach, gentle bronze, pale gold", avoid: "Heavy dark shadows, cool tones, stark liner" },
      blush: { wear: "Soft peachy pink, pale apricot, light coral — applied very lightly", avoid: "Cool pink, heavy blush, dark berry" },
    },
    accentColor: "#F0A888",
    bgColor: "#FFF8F0",
    textColor: "#3A1A05",
  },

  spring_bright: {
    displayName: "Bright Spring",
    season: "spring",
    description: "Vivid, warm, and dazzlingly clear — Bright Spring is the most vivid of all the Springs, carrying colors that are both warm and richly saturated. Your natural clarity calls for colors that match your own aliveness.",
    bestColors: [
      { name: "Vivid coral", hex: "#F05040" }, { name: "Golden yellow", hex: "#F0C020" },
      { name: "Bright turquoise", hex: "#00C0B0" }, { name: "Hot peach", hex: "#F07840" },
      { name: "Bright warm green", hex: "#68C030" }, { name: "Clear red", hex: "#E82020" },
      { name: "Vivid orange", hex: "#F07020" }, { name: "Clear cobalt", hex: "#2060E8" },
      { name: "Bright ivory", hex: "#F8F4D8" }, { name: "Vivid warm pink", hex: "#F04890" },
      { name: "Lemon yellow", hex: "#F0E020" }, { name: "Bright aqua", hex: "#20D0C0" },
    ],
    avoidColors: [
      { name: "Dusty rose", hex: "#C09090" }, { name: "Muted sage", hex: "#909070" },
      { name: "Warm gray", hex: "#A8A098" }, { name: "Dark navy", hex: "#081030" },
      { name: "Black", hex: "#000000" }, { name: "Muted olive", hex: "#787840" },
      { name: "Cool gray", hex: "#9098A8" }, { name: "Deep burgundy", hex: "#600018" },
      { name: "Dusty lavender", hex: "#B0A0C0" }, { name: "Muted camel", hex: "#B89060" },
      { name: "Dark brown", hex: "#401808" }, { name: "Muted coral", hex: "#C09080" },
    ],
    clothes: {
      tops: { wear: "Vivid coral, golden yellow, bright turquoise, clear red, hot peach, vivid warm pink", avoid: "Muted, dusty, or dark colors — you need full saturation to match your vibrancy" },
      bottoms: { wear: "Bright ivory, vivid warm green, clear cobalt, vivid orange", avoid: "Dark navy, black, muted tones that dull your brightness" },
      bagsShoes: { wear: "Bright coral, vivid yellow, clear red, fun bright colors", avoid: "Muted or dark accessories that dim your energy" },
      metals: { wear: "Bright gold, warm brass, vivid rose gold, polished bronze", avoid: "Muted metals, dark iron, cool silver" },
      pairingTips: { wear: "Vivid coral + turquoise is electrifying on you. Bright warm pink + golden yellow is equally stunning. Always go for contrast and saturation." },
    },
    makeup: {
      lips: { wear: "Vivid coral, bright warm red, hot peach, vivid warm pink", avoid: "Muted, dusty, or cool-toned lips" },
      eyes: { wear: "Vivid copper, bright warm brown, vivid gold, clear warm bronzes", avoid: "Muted, dusty, or cool shadows" },
      blush: { wear: "Vivid warm peach, bright coral, clear warm pink — richly applied", avoid: "Cool pink, muted blush, dusty rose" },
    },
    accentColor: "#F05040",
    bgColor: "#FFF8F0",
    textColor: "#3A1A05",
  },

  autumn_warm: {
    displayName: "Warm Autumn",
    season: "autumn",
    description: "Richly warm, earthy, and naturally abundant — Warm Autumn is the golden heart of the Autumn family. Your natural warmth radiates in the richest amber, terracotta, and moss tones of the earth.",
    bestColors: [
      { name: "Warm rust", hex: "#B84020" }, { name: "Golden amber", hex: "#C88020" },
      { name: "Terracotta", hex: "#B86040" }, { name: "Moss green", hex: "#688030" },
      { name: "Camel", hex: "#C09060" }, { name: "Warm tan", hex: "#C8A070" },
      { name: "Burnt orange", hex: "#D06020" }, { name: "Olive gold", hex: "#988030" },
      { name: "Warm brown", hex: "#8B5020" }, { name: "Deep teal", hex: "#287868" },
      { name: "Tomato red", hex: "#C03020" }, { name: "Warm ivory", hex: "#F8F0D0" },
    ],
    avoidColors: [
      { name: "Icy blue", hex: "#B8D0E8" }, { name: "Cool pink", hex: "#E090B0" },
      { name: "Stark white", hex: "#F8F8F8" }, { name: "Electric neon", hex: "#80F000" },
      { name: "Cool gray", hex: "#9098A8" }, { name: "Navy", hex: "#08205A" },
      { name: "Bright purple", hex: "#8030C8" }, { name: "Icy lavender", hex: "#D0C8E8" },
      { name: "Hot pink", hex: "#F01890" }, { name: "Silver", hex: "#B0B8C0" },
      { name: "Cool red", hex: "#C00060" }, { name: "Black", hex: "#000000" },
    ],
    clothes: {
      tops: { wear: "Warm rust, golden amber, terracotta, tomato red, burnt orange, warm ivory", avoid: "Icy blue, cool pink, silver, black — they kill your warmth" },
      bottoms: { wear: "Camel, warm brown, olive, dark moss, warm tan", avoid: "Navy, cool gray, icy or bright colors" },
      bagsShoes: { wear: "Tan leather, warm camel, rich brown, suede, cognac", avoid: "Black, silver hardware, cool-toned accessories" },
      metals: { wear: "Warm gold, bronze, antique gold, copper, brass", avoid: "Silver, platinum, cool white gold" },
      pairingTips: { wear: "Rust + camel is your most natural pairing. Warm amber + moss green creates stunning earth-toned looks. Stay entirely in the warm, earthy spectrum." },
    },
    makeup: {
      lips: { wear: "Terracotta, warm brick red, warm coral, peachy nude, warm brown", avoid: "Cool blue-based reds, berry, cool pink, icy shades" },
      eyes: { wear: "Warm bronze, copper, amber, warm brown, terracotta", avoid: "Cool gray, icy silver, cool lavender" },
      blush: { wear: "Warm peach, apricot, warm terracotta blush", avoid: "Cool pink, cool berry, icy or muted blush" },
    },
    accentColor: "#B84020",
    bgColor: "#FFF4E8",
    textColor: "#3A1A05",
  },

  autumn_soft: {
    displayName: "Soft Autumn",
    season: "autumn",
    description: "Muted, gently warm, and softly glowing — Soft Autumn bridges the warmth of Autumn with the subtlety of Summer. Your natural coloring is delicately warm and harmoniously soft.",
    bestColors: [
      { name: "Muted rust", hex: "#A86040" }, { name: "Warm sage", hex: "#88A070" },
      { name: "Dusty peach", hex: "#D8A080" }, { name: "Camel beige", hex: "#C0A870" },
      { name: "Soft olive", hex: "#909050" }, { name: "Warm mauve", hex: "#B09080" },
      { name: "Dusty coral", hex: "#C09070" }, { name: "Muted teal", hex: "#608878" },
      { name: "Warm beige", hex: "#D0B890" }, { name: "Muted gold", hex: "#B89040" },
      { name: "Rose brown", hex: "#B08878" }, { name: "Warm cream", hex: "#F8F0D8" },
    ],
    avoidColors: [
      { name: "Vivid red", hex: "#E00020" }, { name: "Electric blue", hex: "#0050E8" },
      { name: "True black", hex: "#000000" }, { name: "Pure white", hex: "#FFFFFF" },
      { name: "Hot pink", hex: "#E81890" }, { name: "Bright neon", hex: "#80F000" },
      { name: "Cool lavender", hex: "#B0A0D0" }, { name: "Icy pink", hex: "#F0D0E0" },
      { name: "Bright yellow", hex: "#F0E000" }, { name: "Stark silver", hex: "#C0C8D0" },
      { name: "Deep navy", hex: "#081030" }, { name: "Stark contrast", hex: "#000000" },
    ],
    clothes: {
      tops: { wear: "Muted rust, dusty peach, warm sage, warm mauve, dusty coral, warm cream", avoid: "Vivid, stark, or highly contrasted colors — your warmth is gentle, not bold" },
      bottoms: { wear: "Camel beige, soft olive, warm beige, muted gold brown", avoid: "Black, deep navy, vivid or very cool colors" },
      bagsShoes: { wear: "Soft camel, warm nude, muted tan leather, warm taupe", avoid: "Stark black, cool silver, vivid colors" },
      metals: { wear: "Soft gold, matte bronze, antique warm metals, warm rose gold", avoid: "Bright silver, cool chrome, stark metals" },
      pairingTips: { wear: "Soft sage + warm camel creates beautiful harmony. Dusty peach + muted rust is quietly stunning. Keep everything in a soft, blended warmth." },
    },
    makeup: {
      lips: { wear: "Dusty rose-brown, warm muted coral, soft terracotta, warm nude", avoid: "Vivid red, cool berry, icy shades, very dark colors" },
      eyes: { wear: "Soft bronze, muted copper, warm taupe, dusty warm brown", avoid: "Cool gray, bright colors, heavy black liner" },
      blush: { wear: "Soft warm peach, muted apricot, dusty warm pink", avoid: "Cool or vivid blush shades" },
    },
    accentColor: "#A86040",
    bgColor: "#FFF4E8",
    textColor: "#3A1A05",
  },

  autumn_deep: {
    displayName: "Deep Autumn",
    season: "autumn",
    description: "Rich, dark, and intensely earthy — Deep Autumn carries the most dramatic depths of the Autumn family with natural ease. Your deep coloring handles the richest, most saturated warm tones beautifully.",
    bestColors: [
      { name: "Dark rust", hex: "#902818" }, { name: "Deep olive", hex: "#507028" },
      { name: "Chocolate", hex: "#582010" }, { name: "Forest green", hex: "#285830" },
      { name: "Deep teal", hex: "#186860" }, { name: "Burnt sienna", hex: "#903020" },
      { name: "Dark camel", hex: "#A07030" }, { name: "Deep burgundy", hex: "#600820" },
      { name: "Dark gold", hex: "#A07018" }, { name: "Warm black-brown", hex: "#281008" },
      { name: "Mahogany", hex: "#701018" }, { name: "Dark olive", hex: "#486020" },
    ],
    avoidColors: [
      { name: "Icy pink", hex: "#F0D0E0" }, { name: "Baby blue", hex: "#C0D8F8" },
      { name: "Pale yellow", hex: "#F8F0A0" }, { name: "Mint green", hex: "#B0F0D8" },
      { name: "Lavender", hex: "#D0C0E8" }, { name: "Cool gray", hex: "#9098A8" },
      { name: "Silver", hex: "#B8C0C8" }, { name: "Stark white", hex: "#F8F8F8" },
      { name: "Cool purple", hex: "#8058C8" }, { name: "Electric blue", hex: "#0050E8" },
      { name: "Bright neon", hex: "#80F000" }, { name: "Icy tones", hex: "#D0E0F0" },
    ],
    clothes: {
      tops: { wear: "Dark rust, burnt sienna, forest green, chocolate, deep teal, mahogany", avoid: "Icy pastels, cool tones, silver, pale or light colors" },
      bottoms: { wear: "Chocolate, warm black-brown, dark olive, deep forest green", avoid: "Stark black (too cool), bright or icy colors" },
      bagsShoes: { wear: "Rich chocolate leather, dark cognac, warm dark brown, forest suede", avoid: "Black, cool gray, silver hardware" },
      metals: { wear: "Antique gold, aged bronze, dark copper, oxidized brass", avoid: "Silver, white gold, bright gold, cool metals" },
      pairingTips: { wear: "Deep rust + chocolate brown is your most natural power look. Forest green + dark gold is equally striking. Go deep, warm, and richly layered." },
    },
    makeup: {
      lips: { wear: "Deep warm red, brick red, dark terracotta, warm dark berry", avoid: "Cool berry, icy pink, pale nude" },
      eyes: { wear: "Dark bronze, deep copper, espresso, dark olive, rich warm brown", avoid: "Cool shadows, silver, icy tones, bright colors" },
      blush: { wear: "Warm terracotta, deep warm peach, brick blush — rich and buildable", avoid: "Cool pink, pale blush, icy or muted tones" },
    },
    accentColor: "#902818",
    bgColor: "#FFF4E8",
    textColor: "#3A1A05",
  },
};

export const SUBTYPE_DISPLAY: Record<string, string> = {
  autumn_deep: "Deep Autumn", autumn_soft: "Soft Autumn", autumn_warm: "Warm Autumn",
  spring_bright: "Bright Spring", spring_light: "Light Spring", spring_warm: "Warm Spring",
  summer_cool: "Cool Summer", summer_light: "Light Summer", summer_soft: "Soft Summer",
  winter_bright: "Bright Winter", winter_cool: "Cool Winter", winter_deep: "Deep Winter",
};
