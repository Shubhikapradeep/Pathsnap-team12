# PathSnap evaluator UI

This is a static browser UI. It reads the trip list, predictions, route geometry, warnings, and every metric from the evaluator API. It does not load trip files, reference labels, models, or calculate accuracy.

## Run on the evaluator laptop

From the repository root, start a small static file server:

```bash
cd ui
python3 -m http.server 8080
```

Open <http://localhost:8080>. In **Settings**, enter the API laptop's current address, for example `http://192.168.1.20:8000`, then use **Test connection**. The address is saved in this browser's local storage. Both laptops should be on the same phone hotspot, and the API must be listening on `0.0.0.0:8000` with CORS enabled.

The page calls `GET /health`, `GET /trips`, and `POST /predict/{trip_id}` as defined in [`../docs/API_CONTRACT.md`](../docs/API_CONTRACT.md). The Leaflet library is loaded from its CDN; map tiles come from OpenStreetMap.

## API currently unavailable

The API service itself is maintained separately. This checkout contains the API contract and example payloads, but no local server implementation. The UI does not substitute the example JSON for live API responses. When the service is running, set its address in Settings and reload the trip list by reloading the page.
