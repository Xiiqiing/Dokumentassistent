variable "base_name" {
  description = "Base name used to derive resource names"
  type        = string
  default     = "kudocassist"
}

variable "location" {
  description = "Azure region for all resources"
  type        = string
  default     = "northeurope"
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
  default     = "rg-kudocassist"
}

variable "image_tag" {
  description = "Container image tag"
  type        = string
  default     = "latest"
}

variable "llm_provider" {
  description = "LLM provider (ollama, openai, azure_openai, google_genai, etc.)"
  type        = string
  default     = "google_genai"
}

variable "embedding_provider" {
  description = "Embedding provider (local, openai, azure_openai, google_genai, etc.)"
  type        = string
  default     = "local"
}

variable "generation_model" {
  description = "Generation model name"
  type        = string
  default     = "gemini-2.5-flash"
}

variable "google_api_key" {
  description = "Google API key (required when llm_provider is google_genai)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "openai_api_key" {
  description = "OpenAI API key (required when llm_provider is openai)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "azure_openai_api_key" {
  description = "Azure OpenAI API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "azure_openai_endpoint" {
  description = "Azure OpenAI endpoint URL"
  type        = string
  default     = ""
}

variable "azure_openai_deployment" {
  description = "Azure OpenAI deployment name"
  type        = string
  default     = ""
}
