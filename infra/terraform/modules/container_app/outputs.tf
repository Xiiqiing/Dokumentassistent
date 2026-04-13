output "id" {
  value = azurerm_container_app.this.id
}

output "fqdn" {
  description = "Ingress FQDN (internal or external depending on configuration)"
  value       = azurerm_container_app.this.ingress[0].fqdn
}
