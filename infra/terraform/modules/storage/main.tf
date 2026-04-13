# ---------------------------------------------------------------------------
# Storage module: storage account, file share, and ACA environment binding.
# Used for stateful workloads that need persistent volumes (e.g. Qdrant).
# ---------------------------------------------------------------------------

locals {
  clean_name = replace(var.base_name, "-", "")
}

resource "azurerm_storage_account" "main" {
  name                     = "${local.clean_name}stor"
  location                 = var.location
  resource_group_name      = var.resource_group_name
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_storage_share" "this" {
  name               = var.share_name
  storage_account_id = azurerm_storage_account.main.id
  quota              = var.quota_gb
}

resource "azurerm_container_app_environment_storage" "this" {
  name                         = var.env_storage_name
  container_app_environment_id = var.container_app_environment_id
  account_name                 = azurerm_storage_account.main.name
  access_key                   = azurerm_storage_account.main.primary_access_key
  share_name                   = azurerm_storage_share.this.name
  access_mode                  = "ReadWrite"
}
