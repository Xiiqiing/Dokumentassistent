output "env_storage_name" {
  description = "ACA environment storage binding name (use as volume.storage_name)"
  value       = azurerm_container_app_environment_storage.this.name
}
