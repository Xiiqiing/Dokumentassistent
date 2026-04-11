# ---------------------------------------------------------------------------
# KU Doc Assistant - Azure Container Apps deployment (Terraform)
# ---------------------------------------------------------------------------

locals {
  clean_name = replace(var.base_name, "-", "")
}

# ---------------------------------------------------------------------------
# Resource Group
# ---------------------------------------------------------------------------

resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
}

# ---------------------------------------------------------------------------
# Log Analytics
# ---------------------------------------------------------------------------

resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.base_name}-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

# ---------------------------------------------------------------------------
# Container App Environment
# ---------------------------------------------------------------------------

resource "azurerm_container_app_environment" "main" {
  name                       = "${var.base_name}-env"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
}

# ---------------------------------------------------------------------------
# Azure Container Registry
# ---------------------------------------------------------------------------

resource "azurerm_container_registry" "main" {
  name                = "${local.clean_name}acr"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "Basic"
  admin_enabled       = true
}

# ---------------------------------------------------------------------------
# Storage Account + File Share for Qdrant persistence
# ---------------------------------------------------------------------------

resource "azurerm_storage_account" "main" {
  name                     = "${local.clean_name}stor"
  location                 = azurerm_resource_group.main.location
  resource_group_name      = azurerm_resource_group.main.name
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_storage_share" "qdrant" {
  name               = "qdrant-data"
  storage_account_id = azurerm_storage_account.main.id
  quota              = 5
}

resource "azurerm_container_app_environment_storage" "qdrant" {
  name                         = "qdrantstorage"
  container_app_environment_id = azurerm_container_app_environment.main.id
  account_name                 = azurerm_storage_account.main.name
  access_key                   = azurerm_storage_account.main.primary_access_key
  share_name                   = azurerm_storage_share.qdrant.name
  access_mode                  = "ReadWrite"
}

# ---------------------------------------------------------------------------
# Qdrant Container App
# ---------------------------------------------------------------------------

resource "azurerm_container_app" "qdrant" {
  name                         = "${var.base_name}-qdrant"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  ingress {
    external_enabled = false
    target_port      = 6333
    transport        = "http"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 1
    max_replicas = 1

    container {
      name   = "qdrant"
      image  = "qdrant/qdrant:latest"
      cpu    = 0.5
      memory = "1Gi"

      volume_mounts {
        name = "qdrant-vol"
        path = "/qdrant/storage"
      }
    }

    volume {
      name         = "qdrant-vol"
      storage_name = azurerm_container_app_environment_storage.qdrant.name
      storage_type = "AzureFile"
    }
  }
}

# ---------------------------------------------------------------------------
# API Container App
# ---------------------------------------------------------------------------

resource "azurerm_container_app" "api" {
  name                         = "${var.base_name}-api"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  secret {
    name  = "acr-password"
    value = azurerm_container_registry.main.admin_password
  }

  secret {
    name  = "google-api-key"
    value = var.google_api_key
  }

  secret {
    name  = "openai-api-key"
    value = var.openai_api_key
  }

  secret {
    name  = "azure-openai-api-key"
    value = var.azure_openai_api_key
  }

  registry {
    server               = azurerm_container_registry.main.login_server
    username             = azurerm_container_registry.main.admin_username
    password_secret_name = "acr-password"
  }

  ingress {
    external_enabled = false
    target_port      = 8000
    transport        = "http"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 1
    max_replicas = 3

    container {
      name   = "api"
      image  = "${azurerm_container_registry.main.login_server}/${var.base_name}-api:${var.image_tag}"
      cpu    = 1.0
      memory = "2Gi"

      env {
        name  = "QDRANT_URL"
        value = "http://${azurerm_container_app.qdrant.ingress[0].fqdn}"
      }
      env {
        name  = "LLM_PROVIDER"
        value = var.llm_provider
      }
      env {
        name  = "EMBEDDING_PROVIDER"
        value = var.embedding_provider
      }
      env {
        name  = "GENERATION_MODEL"
        value = var.generation_model
      }
      env {
        name        = "GOOGLE_API_KEY"
        secret_name = "google-api-key"
      }
      env {
        name        = "OPENAI_API_KEY"
        secret_name = "openai-api-key"
      }
      env {
        name        = "AZURE_OPENAI_API_KEY"
        secret_name = "azure-openai-api-key"
      }
      env {
        name  = "AZURE_OPENAI_ENDPOINT"
        value = var.azure_openai_endpoint
      }
      env {
        name  = "AZURE_OPENAI_DEPLOYMENT"
        value = var.azure_openai_deployment
      }
      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }

      readiness_probe {
        transport        = "HTTP"
        path             = "/health/ready"
        port             = 8000
        initial_delay    = 30
        interval_seconds = 15
      }

      liveness_probe {
        transport        = "HTTP"
        path             = "/health"
        port             = 8000
        interval_seconds = 30
      }
    }
  }
}

# ---------------------------------------------------------------------------
# UI Container App (Streamlit)
# ---------------------------------------------------------------------------

resource "azurerm_container_app" "ui" {
  name                         = "${var.base_name}-ui"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  secret {
    name  = "acr-password"
    value = azurerm_container_registry.main.admin_password
  }

  registry {
    server               = azurerm_container_registry.main.login_server
    username             = azurerm_container_registry.main.admin_username
    password_secret_name = "acr-password"
  }

  ingress {
    external_enabled = true
    target_port      = 8501
    transport        = "http"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 1
    max_replicas = 3

    container {
      name    = "ui"
      image   = "${azurerm_container_registry.main.login_server}/${var.base_name}-ui:${var.image_tag}"
      cpu     = 0.5
      memory  = "1Gi"
      command = ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]

      env {
        name  = "API_BASE_URL"
        value = "http://${azurerm_container_app.api.ingress[0].fqdn}"
      }
    }
  }
}
