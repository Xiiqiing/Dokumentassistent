# ---------------------------------------------------------------------------
# Generic Container App module.
# Used for qdrant, api, ui — all three differ only in image, port, env,
# secrets, and probes.
# ---------------------------------------------------------------------------

resource "azurerm_container_app" "this" {
  name                         = var.name
  container_app_environment_id = var.container_app_environment_id
  resource_group_name          = var.resource_group_name
  revision_mode                = "Single"

  dynamic "secret" {
    for_each = var.secrets
    content {
      name  = secret.key
      value = secret.value
    }
  }

  dynamic "registry" {
    for_each = var.registry == null ? [] : [var.registry]
    content {
      server               = registry.value.server
      username             = registry.value.username
      password_secret_name = registry.value.password_secret_name
    }
  }

  ingress {
    external_enabled = var.external_ingress
    target_port      = var.target_port
    transport        = "http"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    container {
      name    = var.container_name
      image   = var.image
      cpu     = var.cpu
      memory  = var.memory
      command = length(var.command) > 0 ? var.command : null

      dynamic "env" {
        for_each = var.env
        content {
          name        = env.value.name
          value       = env.value.value
          secret_name = env.value.secret_name
        }
      }

      dynamic "readiness_probe" {
        for_each = var.readiness_probe == null ? [] : [var.readiness_probe]
        content {
          transport        = "HTTP"
          path             = readiness_probe.value.path
          port             = readiness_probe.value.port
          initial_delay    = readiness_probe.value.initial_delay
          interval_seconds = readiness_probe.value.interval_seconds
        }
      }

      dynamic "liveness_probe" {
        for_each = var.liveness_probe == null ? [] : [var.liveness_probe]
        content {
          transport        = "HTTP"
          path             = liveness_probe.value.path
          port             = liveness_probe.value.port
          interval_seconds = liveness_probe.value.interval_seconds
        }
      }

      dynamic "volume_mounts" {
        for_each = var.volume_mounts
        content {
          name = volume_mounts.value.name
          path = volume_mounts.value.path
        }
      }
    }

    dynamic "volume" {
      for_each = var.volumes
      content {
        name         = volume.value.name
        storage_name = volume.value.storage_name
        storage_type = "AzureFile"
      }
    }
  }
}
