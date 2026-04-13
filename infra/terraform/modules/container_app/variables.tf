variable "name" {
  description = "Container App resource name"
  type        = string
}

variable "container_name" {
  description = "Name of the container inside the app"
  type        = string
}

variable "container_app_environment_id" {
  description = "ID of the parent ACA environment"
  type        = string
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
}

variable "image" {
  description = "Container image (with tag)"
  type        = string
}

variable "cpu" {
  description = "vCPU allocation"
  type        = number
  default     = 0.5
}

variable "memory" {
  description = "Memory allocation (e.g. 1Gi)"
  type        = string
  default     = "1Gi"
}

variable "target_port" {
  description = "Container port for ingress"
  type        = number
}

variable "external_ingress" {
  description = "Whether ingress is exposed publicly"
  type        = bool
  default     = false
}

variable "min_replicas" {
  type    = number
  default = 1
}

variable "max_replicas" {
  type    = number
  default = 1
}

variable "command" {
  description = "Container entrypoint override"
  type        = list(string)
  default     = []
}

variable "secrets" {
  description = "Map of secret name => value, exposed to the container"
  type        = map(string)
  default     = {}
  sensitive   = true
}

variable "registry" {
  description = "Optional private registry credentials"
  type = object({
    server               = string
    username             = string
    password_secret_name = string
  })
  default = null
}

variable "env" {
  description = "Environment variables. Each entry sets either value or secret_name."
  type = list(object({
    name        = string
    value       = optional(string)
    secret_name = optional(string)
  }))
  default = []
}

variable "readiness_probe" {
  description = "Optional HTTP readiness probe"
  type = object({
    path             = string
    port             = number
    initial_delay    = optional(number)
    interval_seconds = optional(number)
  })
  default = null
}

variable "liveness_probe" {
  description = "Optional HTTP liveness probe"
  type = object({
    path             = string
    port             = number
    interval_seconds = optional(number)
  })
  default = null
}

variable "volume_mounts" {
  description = "Volume mounts for the container"
  type = list(object({
    name = string
    path = string
  }))
  default = []
}

variable "volumes" {
  description = "Volumes attached to the template (referencing ACA env storage)"
  type = list(object({
    name         = string
    storage_name = string
  }))
  default = []
}
