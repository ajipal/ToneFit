export type Season = "spring" | "summer" | "autumn" | "winter";

export type SubType =
  | "spring_warm" | "spring_light" | "spring_bright"
  | "summer_cool" | "summer_light" | "summer_soft"
  | "autumn_warm" | "autumn_soft" | "autumn_deep"
  | "winter_cool" | "winter_deep" | "winter_bright";

export interface PredictionResult {
  season: Season;
  subtype: SubType;
  season_confidence: number;
  subtype_confidence: number;
  top3: Array<{ subtype: SubType; confidence: number }>;
}

export interface UserProfile {
  name: string;
  style: string;
}
