
const map = L.map("map").setView([27.036, 88.262], 8);
let selectedMarker = null;
let selectedLocation = null;
let previewUrl = null;

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19
}).addTo(map);

const $ = (id) => document.getElementById(id);

async function checkApi() {
    try {
        const response = await fetch("/api/v1/health");
        if (!response.ok) throw new Error("API unavailable");
        $("api-status").textContent = "Connected";
    } catch {
        $("api-status").textContent = "Connection unavailable";
    }
}

function setLocation(name, lat, lon) {
    selectedLocation = { name, lat, lon };
    $("location-stat").textContent = name;
    $("coordinates").textContent = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
    $("location-message").textContent = "Location selected. Satellite imagery retrieval is not connected yet.";

    if (selectedMarker) map.removeLayer(selectedMarker);
    selectedMarker = L.marker([lat, lon]).addTo(map).bindPopup(name).openPopup();
    map.setView([lat, lon], 12);
}

map.on("click", (event) => {
    const { lat, lng } = event.latlng;
    setLocation("Selected map point", lat, lng);
});

$("location-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = $("location-input").value.trim();
    if (!query) return;

    $("location-message").textContent = "Searching for location…";

    try {
        // OpenStreetMap Nominatim geocoding service
        const url = new URL("https://nominatim.openstreetmap.org/search");
        url.searchParams.set("q", query);
        url.searchParams.set("format", "jsonv2");
        url.searchParams.set("limit", "1");

        const response = await fetch(url);
        if (!response.ok) throw new Error("Location search failed");

        const results = await response.json();
        if (!results.length) {
            $("location-message").textContent = "No matching location found. Try a more specific place name.";
            return;
        }

        const result = results[0];
        setLocation(result.display_name, Number(result.lat), Number(result.lon));
        $("location-message").textContent = "Location found. Satellite imagery integration is the next step.";
    } catch (error) {
        $("location-message").textContent =
            "Location search failed. Check your internet connection and try again.";
    }
});

$("image-input").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) return;

    $("image-stat").textContent = "Selected";
    $("file-name").textContent = file.name;

    $("drop-area").classList.add("hidden");
    $("preview-area").classList.remove("hidden");

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;

    if (file.type.startsWith("image/")) {
        previewUrl = URL.createObjectURL(file);
        $("image-preview").src = previewUrl;
        $("image-preview").classList.remove("hidden");
    } else {
        $("image-preview").removeAttribute("src");
        $("image-preview").classList.add("hidden");
        $("file-name").textContent += " — selected; HDF5 preview is not available in the browser.";
    }

    $("prediction-message").textContent =
        "Image selected. Prediction is not connected to the model API yet.";
});

$("remove-image").addEventListener("click", () => {
    $("image-input").value = "";
    $("preview-area").classList.add("hidden");
    $("drop-area").classList.remove("hidden");
    $("image-stat").textContent = "No image";

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    $("image-preview").removeAttribute("src");
});

checkApi();
