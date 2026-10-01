export interface GenerationSettings {
  max_new_tokens: number;
  temperature: number;
  top_k: number;
  top_p?: number;
  seed?: number;
}

export interface GenerateRequest {
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_k: number;
  top_p?: number;
  seed?: number;
}

export interface GenerateResponse {
  prompt: string;
  generated_text: string;
  settings: GenerationSettings;
  runtime_ms: number;
  model_version: string;
}

export interface CompareResultItem {
  name: string;
  settings: GenerationSettings;
  generated_text: string;
  runtime_ms: number;
}

export interface CompareRequest {
  prompt: string;
  max_new_tokens?: number;
  settings_list?: GenerationSettings[];
}

export interface CompareResponse {
  prompt: string;
  results: CompareResultItem[];
}

export interface ModelStatusResponse {
  status: string;
  model_name: string;
  version: string;
  vocab_size: number;
  device: string;
  parameters_count?: number;
}

export interface ModelMetricsResponse {
  status: string;
  metrics: {
    training_duration_seconds?: number;
    epochs_completed?: number;
    final_loss?: number;
    best_train_loss?: number;
    best_val_loss?: number;
    perplexity?: number;
    train_samples?: number;
    val_samples?: number;
    token_accuracy?: number;
    [key: string]: any;
  };
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  model_loaded: boolean;
  version: string;
}
