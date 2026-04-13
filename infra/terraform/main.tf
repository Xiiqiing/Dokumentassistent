# ---------------------------------------------------------------------------
# KU Doc Assistant - root composition.
# Wires the platform, storage, and container_app modules together.
# ---------------------------------------------------------------------------

module "platform" {
  source              = "./modules/platform"
  base_name           = var.base_name
  location            = var.location
  resource_group_name = var.resource_group_name
}

module "qdrant_storage" {
  source                       = "./modules/storage"
  base_name                    = var.base_name
  location                     = module.platform.location
  resource_group_name          = module.platform.resource_group_name
  container_app_environment_id = module.platform.container_app_environment_id
  share_name                   = "qdrant-data"
  env_storage_name             = "qdrantstorage"
  quota_gb                     = 5
}

locals {
  acr_registry = {
    server               = module.platform.acr_login_server
    username             = module.platform.acr_admin_username
    password_secret_name = "acr-password"
  }

  api_secrets = {
    "acr-password"         = module.platform.acr_admin_password
    "google-api-key"       = var.google_api_key
    "openai-api-key"       = var.openai_api_key
    "azure-openai-api-key" = var.azure_openai_api_key
  }

  ui_secrets = {
    "acr-password" = module.platform.acr_admin_password
  }
}

# ---------------------------------------------------------------------------
# Qdrant — internal stateful service backed by ACA file-share volume
# ---------------------------------------------------------------------------

module "qdrant" {
  source = "./modules/container_app"

  name                         = "${var.base_name}-qdrant"
  container_name               = "qdrant"
  container_app_environment_id = module.platform.container_app_environment_id
  resource_group_name          = module.platform.resource_group_name

  image            = "qdrant/qdrant:latest"
  cpu              = 0.5
  memory           = "1Gi"
  target_port      = 6333
  external_ingress = false

  volume_mounts = [
    { name = "qdrant-vol", path = "/qdrant/storage" }
  ]
  volumes = [
    { name = "qdrant-vol", storage_name = module.qdrant_storage.env_storage_name }
  ]
}

# ---------------------------------------------------------------------------
# API — internal FastAPI backend
# ---------------------------------------------------------------------------

module "api" {
  source = "./modules/container_app"

  name                         = "${var.base_name}-api"
  container_name               = "api"
  container_app_environment_id = module.platform.container_app_environment_id
  resource_group_name          = module.platform.resource_group_name

  image            = "${module.platform.acr_login_server}/${var.base_name}-api:${var.image_tag}"
  cpu              = 1.0
  memory           = "2Gi"
  target_port      = 8000
  external_ingress = false
  max_replicas     = 3

  registry = local.acr_registry
  secrets  = local.api_secrets

  env = [
    { name = "QDRANT_URL", value = "http://${module.qdrant.fqdn}" },
    { name = "LLM_PROVIDER", value = var.llm_provider },
    { name = "EMBEDDING_PROVIDER", value = var.embedding_provider },
    { name = "GENERATION_MODEL", value = var.generation_model },
    { name = "GOOGLE_API_KEY", secret_name = "google-api-key" },
    { name = "OPENAI_API_KEY", secret_name = "openai-api-key" },
    { name = "AZURE_OPENAI_API_KEY", secret_name = "azure-openai-api-key" },
    { name = "AZURE_OPENAI_ENDPOINT", value = var.azure_openai_endpoint },
    { name = "AZURE_OPENAI_DEPLOYMENT", value = var.azure_openai_deployment },
    { name = "LOG_LEVEL", value = "INFO" },
  ]

  readiness_probe = {
    path             = "/health/ready"
    port             = 8000
    initial_delay    = 30
    interval_seconds = 15
  }

  liveness_probe = {
    path             = "/health"
    port             = 8000
    interval_seconds = 30
  }
}

# ---------------------------------------------------------------------------
# UI — public Streamlit frontend
# ---------------------------------------------------------------------------

module "ui" {
  source = "./modules/container_app"

  name                         = "${var.base_name}-ui"
  container_name               = "ui"
  container_app_environment_id = module.platform.container_app_environment_id
  resource_group_name          = module.platform.resource_group_name

  image            = "${module.platform.acr_login_server}/${var.base_name}-ui:${var.image_tag}"
  cpu              = 0.5
  memory           = "1Gi"
  target_port      = 8501
  external_ingress = true
  max_replicas     = 3

  registry = local.acr_registry
  secrets  = local.ui_secrets

  command = [
    "streamlit", "run", "src/ui/app.py",
    "--server.port=8501", "--server.address=0.0.0.0",
    "--server.headless=true", "--browser.gatherUsageStats=false",
  ]

  env = [
    { name = "API_BASE_URL", value = "http://${module.api.fqdn}" },
  ]
}
