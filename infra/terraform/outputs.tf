output "ui_url" {
  description = "Public URL of the Streamlit UI"
  value       = "https://${azurerm_container_app.ui.ingress[0].fqdn}"
}

output "api_internal_url" {
  description = "Internal URL of the FastAPI backend"
  value       = "http://${azurerm_container_app.api.ingress[0].fqdn}"
}

output "acr_login_server" {
  description = "Azure Container Registry login server"
  value       = azurerm_container_registry.main.login_server
}
