// ======================
// CONFIGURE YOUR ENDPOINT
// ======================
const API_ENDPOINT = "http://127.0.0.1:8000/predict";
// Cloud Run version:
// const API_ENDPOINT = "https://smart-recipe-app-249886303998.northamerica-northeast2.run.app/predict";

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("endpointText").textContent = API_ENDPOINT;
});

// ======================
// Example payload loader
// ======================
function examplePayload() {
  return {
    ingredients: ["tomato", "onion", "olive oil", "garlic", "basil"],
    diet: "none",
    top_k: 3,
    notes: ""
  };
}

function loadExample() {
  const ex = examplePayload();
  document.getElementById("ingredients").value = ex.ingredients.join(", ");
  document.getElementById("diet").value = ex.diet;
  document.getElementById("topk").value = ex.top_k;
  document.getElementById("notes").value = ex.notes;
  renderJSON("reqBox", ex);
}

// ======================
// Build payload
// ======================
function buildPayloadFromForm() {
  const ingredientsRaw = document.getElementById("ingredients").value.trim();
  const ingredients = ingredientsRaw.length
    ? ingredientsRaw.split(",").map(s => s.trim()).filter(Boolean)
    : [];

  return {
    ingredients,
    diet: document.getElementById("diet").value,
    top_k: parseInt(document.getElementById("topk").value || "3", 10),
    notes: document.getElementById("notes").value.trim()
  };
}

function renderJSON(id, obj) {
  document.getElementById(id).textContent = JSON.stringify(obj, null, 2);
}

// ======================
// API REQUEST
// ======================
async function sendPredict() {
  const body = buildPayloadFromForm();
  renderJSON("reqBox", body);
  renderJSON("resBox", { status: "Sending..." });

  try {
    const res = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    const text = await res.text();

    try {
      const json = JSON.parse(text);
      renderJSON("resBox", json);
    } catch (_) {
      document.getElementById("resBox").textContent = text;
    }

  } catch (err) {
    document.getElementById("resBox").textContent = "Request failed: " + err;
  }
}
