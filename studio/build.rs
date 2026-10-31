fn main() {
    // Hinweis: ein eigenes app.manifest ueber WindowsAttributes laesst die App
    // hier sofort beenden (ohne Fehlermeldung). Die DPI-Korrektur ist deshalb
    // offen -- siehe RECHERCHE/Notizen. Bis dahin Standard-Manifest.
    tauri_build::build()
}
