variable "base_name" {
  description = "Base name used to derive the storage account name"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
}

variable "resource_group_name" {
  description = "Resource group hosting the storage account"
  type        = string
}

variable "container_app_environment_id" {
  description = "ID of the Container App Environment to bind storage to"
  type        = string
}

variable "share_name" {
  description = "Name of the file share"
  type        = string
}

variable "env_storage_name" {
  description = "Name of the ACA environment storage binding"
  type        = string
}

variable "quota_gb" {
  description = "File share quota in GB"
  type        = number
  default     = 5
}
