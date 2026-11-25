
const API = "http://127.0.0.1:8000";

let chart;
let modelChart;

const $ = (id) => document.getElementById(id);


// =====================================
// TODAY'S DATE
// =====================================

$("today").textContent = new Date().toLocaleDateString(undefined, {
  weekday: "short",
  month: "short",
  day: "numeric",
  year: "numeric"
});


// =====================================
// API REQUEST HELPER
// =====================================

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    }
  });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;

    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {}

    throw new Error(message);
  }

  return response.json();
}


// =====================================
// ESCAPE HTML
// =====================================

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[c]));
}


// =====================================
// SIDEBAR NAVIGATION
// =====================================

const navLinks = document.querySelectorAll(".sidebar nav a");

const navSections = Array.from(navLinks)
  .map(link => {
    const targetId = link.getAttribute("href");

    if (!targetId || !targetId.startsWith("#")) {
      return null;
    }

    return document.querySelector(targetId);
  })
  .filter(section => section !== null);


function setActiveNav(sectionId) {
  navLinks.forEach(link => {
    const targetId = link.getAttribute("href");

    link.classList.toggle(
      "active",
      targetId === `#${sectionId}`
    );
  });
}


// Sidebar click navigation
navLinks.forEach(link => {
  link.addEventListener("click", () => {
    const targetId = link.getAttribute("href");

    if (!targetId || !targetId.startsWith("#")) return;

    const sectionId = targetId.substring(1);

    setActiveNav(sectionId);
  });
});


// Update active navigation while scrolling
let scrollTicking = false;

window.addEventListener("scroll", () => {
  if (scrollTicking) return;

  scrollTicking = true;

  window.requestAnimationFrame(() => {
    let currentSection = null;

    navSections.forEach(section => {
      const position = section.getBoundingClientRect().top;

      if (position <= 160) {
        currentSection = section;
      }
    });

    if (currentSection) {
      setActiveNav(currentSection.id);
    }

    scrollTicking = false;
  });

}, { passive: true });


// Set active navigation on page load
function initializeNavigation() {
  const hash = window.location.hash;

  if (hash) {
    const target = document.querySelector(hash);

    if (target && navSections.includes(target)) {
      setActiveNav(target.id);
      return;
    }
  }

  if (navSections.length > 0) {
    setActiveNav(navSections[0].id);
  }
}

initializeNavigation();


// =====================================
// LOAD ACTIVITY TYPES
// =====================================

async function loadTypes() {
  const data = await request("/api/activities/types");

  $("activity").innerHTML = data.activities
    .map(x => `<option value="${escapeHtml(x)}">${escapeHtml(x)}</option>`)
    .join("");
}


// =====================================
// LOAD DASHBOARD
// =====================================

async function loadDashboard() {
  const [analytics, activities] = await Promise.all([
    request("/api/analytics"),
    request("/api/activities?limit=50")
  ]);

  $("totalActivities").textContent =
    analytics.total_activities;

  const mins = Math.round(analytics.total_minutes);

  $("totalTime").textContent =
    `${Math.floor(mins / 60)}h ${mins % 60}m`;

  renderChart(analytics.duration_by_activity);

  renderHistory(activities);
}


// =====================================
// ACTIVITY DISTRIBUTION CHART
// =====================================

function renderChart(data) {
  const labels = Object.keys(data);
  const values = Object.values(data);

  const activityColors = {
    Sleep: "#8B5CF6",
    Work: "#3B82F6",
    Eating: "#F59E0B",
    Exercise: "#22C55E",
    Entertainment: "#EC4899",
    Study: "#06B6D4",
    Travel: "#F97316",
    Break: "#A3E635"
  };

  const colors = labels.map(
    activity => activityColors[activity] || "#94A3B8"
  );

  if (chart) chart.destroy();

  chart = new Chart($("activityChart"), {
    type: "doughnut",

    data: {
      labels,

      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: "#121a2e",
        borderWidth: 3,
        hoverOffset: 12,
        hoverBorderColor: "#ffffff"
      }]
    },

    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "62%",

      plugins: {
        legend: {
          display: true,
          position: "right",

          labels: {
            color: "#dce6f7",
            padding: 18,
            usePointStyle: true,
            pointStyle: "circle",

            font: {
              size: 12,
              weight: "500"
            }
          }
        },

        tooltip: {
          callbacks: {
            label: function(context) {
              return ` ${context.label}: ${context.parsed} minutes`;
            }
          }
        }
      }
    }
  });
}


// =====================================
// RENDER ACTIVITY HISTORY
// =====================================

function renderHistory(items) {
  const body = $("historyBody");

  if (!items.length) {
    body.innerHTML =
      '<tr><td colspan="6">No activities yet.</td></tr>';

    return;
  }

  body.innerHTML = items.map(x => `
    <tr>
      <td>${escapeHtml(x.activity)}</td>
      <td>${x.duration} min</td>
      <td>${x.energy_level}/5</td>
      <td>${x.priority}/5</td>
      <td>${new Date(x.timestamp).toLocaleString()}</td>

      <td>
        <button class="delete" data-delete="${x.id}">
          Delete
        </button>
      </td>
    </tr>
  `).join("");
}


// =====================================
// SAVE ACTIVITY
// =====================================

$("activityForm").addEventListener("submit", async (event) => {
  event.preventDefault();

  $("formMessage").textContent = "Saving...";

  try {
    await request("/api/activities", {
      method: "POST",

      body: JSON.stringify({
        activity: $("activity").value,
        duration: Number($("duration").value),
        energy_level: Number($("energy").value),
        priority: Number($("priority").value)
      })
    });

    $("formMessage").textContent =
      "Activity saved successfully.";

    await loadDashboard();
    await loadDailyInsights();

  } catch (error) {
    $("formMessage").textContent = error.message;
  }
});


// =====================================
// DELETE ACTIVITY
// =====================================

$("historyBody").addEventListener("click", async (event) => {
  const id = event.target.dataset.delete;

  if (!id) return;

  if (!confirm("Delete this activity?")) return;

  try {
    await request(`/api/activities/${id}`, {
      method: "DELETE"
    });

    await loadDashboard();
    await loadDailyInsights();

  } catch (error) {
    alert(error.message);
  }
});


// =====================================
// RENDER TOP 3 PREDICTIONS
// =====================================

function renderTopPredictions(predictions) {
  const container = $("topPredictions");

  if (!container) return;

  if (!predictions || predictions.length === 0) {
    container.innerHTML = `
      <p class="muted">
        Top predictions are not available yet.
        Add at least 12 activities.
      </p>
    `;

    return;
  }

  container.innerHTML = predictions.slice(0, 3).map((item, index) => {
    const confidence = Number(item.confidence);

    const percentage = Number.isFinite(confidence)
      ? Math.min(100, Math.max(0, confidence * 100))
      : 0;

    const rankLabels = ["1", "2", "3"];

    return `
      <div class="top-prediction-item">

        <div class="top-prediction-info">

          <span class="prediction-rank">
            ${rankLabels[index]}
          </span>

          <strong>
            ${escapeHtml(item.activity)}
          </strong>

          <span class="prediction-percentage">
            ${percentage.toFixed(2)}%
          </span>

        </div>

        <div class="prediction-progress">
          <div
            class="prediction-progress-fill"
            style="width: ${percentage}%">
          </div>
        </div>

      </div>
    `;
  }).join("");
}


// =====================================
// TRANSFORMER PREDICTION
// =====================================

$("predictBtn").addEventListener("click", async () => {
  const button = $("predictBtn");

  button.disabled = true;

  $("predictedActivity").textContent =
    "Analysing your routine...";

  $("suggestion").textContent =
    "Transformer is analysing your recent activity sequence.";

  $("confidence").textContent = "";

  renderTopPredictions([]);

  try {
    const result = await request("/api/predict");

    $("predictedActivity").textContent =
      result.predicted_activity;

    $("suggestion").textContent =
      result.suggestion;

    $("confidence").textContent =
      Number(result.confidence) > 0
        ? `Model confidence: ${(Number(result.confidence) * 100).toFixed(2)}%`
        : "Personalized model prediction not available yet.";

    renderTopPredictions(result.top_predictions);

    await loadPredictionHistory();

  } catch (error) {
    $("predictedActivity").textContent =
      "Prediction unavailable";

    $("suggestion").textContent = error.message;

    $("confidence").textContent = "";

    renderTopPredictions([]);

  } finally {
    button.disabled = false;
  }
});


// =====================================
// LOAD PREDICTION HISTORY
// =====================================

async function loadPredictionHistory() {
  const body = $("predictionHistoryBody");
  const message = $("predictionHistoryMessage");

  try {
    message.textContent = "Loading prediction history...";

    const records = await request("/api/predictions?limit=20");

    if (!records.length) {
      body.innerHTML =
        '<tr><td colspan="4">No prediction records available.</td></tr>';

      message.textContent = "No prediction history found.";

      return;
    }

    body.innerHTML = records.map(item => `
      <tr>
        <td>${escapeHtml(item.predicted_activity)}</td>

        <td>
          ${(Number(item.confidence) * 100).toFixed(1)}%
        </td>

        <td>
          ${escapeHtml(item.suggestion || "No suggestion available")}
        </td>

        <td>
          ${new Date(item.created_at).toLocaleString()}
        </td>
      </tr>
    `).join("");

    message.textContent =
      `Showing ${records.length} prediction records.`;

  } catch (error) {
    body.innerHTML =
      '<tr><td colspan="4">Unable to load prediction history.</td></tr>';

    message.textContent = error.message;
  }
}


// =====================================
// LOAD DAILY INSIGHTS
// =====================================

async function loadDailyInsights() {
  const message = $("dailyInsightsMessage");

  try {
    message.textContent =
      "Analysing your last 24 hours...";

    const data = await request("/api/daily-insights");

    $("insightTotalActivities").textContent =
      data.total_activities ?? 0;

    $("insightTotalMinutes").textContent =
      `${Math.round(data.total_minutes ?? 0)} min`;

    $("insightAverageEnergy").textContent =
      Number(data.average_energy ?? 0).toFixed(2) + "/5";

    $("insightHighPriority").textContent =
      data.high_priority_count ?? 0;

    $("insightMostCommon").textContent =
      data.most_common_activity || "No activity yet";


    const insightsList = $("dailyInsightsList");

    const recommendationsList =
      $("dailyRecommendationsList");

    insightsList.innerHTML = "";

    if (data.insights && data.insights.length > 0) {
      data.insights.forEach(item => {
        const li = document.createElement("li");

        li.textContent = item;

        insightsList.appendChild(li);
      });

    } else {
      insightsList.innerHTML =
        "<li>No insights available yet.</li>";
    }


    recommendationsList.innerHTML = "";

    if (data.recommendations && data.recommendations.length > 0) {
      data.recommendations.forEach(item => {
        const li = document.createElement("li");

        li.textContent = item;

        recommendationsList.appendChild(li);
      });

    } else {
      recommendationsList.innerHTML =
        "<li>No recommendations available yet.</li>";
    }

    message.textContent =
      "Daily insights updated successfully.";

  } catch (error) {
    message.textContent =
      `Could not load daily insights: ${error.message}`;

    $("insightMostCommon").textContent =
      "Unavailable";

    $("dailyInsightsList").innerHTML =
      "<li>Unable to retrieve insights from the backend.</li>";

    $("dailyRecommendationsList").innerHTML =
      "<li>Please check whether the backend is running.</li>";
  }
}


// =====================================
// REFRESH DAILY INSIGHTS
// =====================================

$("dailyInsightsRefreshBtn").addEventListener("click", () => {
  loadDailyInsights();
});


// =====================================
// LOAD MODEL PERFORMANCE
// =====================================

async function loadModelPerformance() {
  const message = $("modelPerformanceMessage");

  try {
    message.textContent =
      "Loading model performance...";

    const data = await request("/api/model-performance");

    const transformerAccuracy =
      Number(data.transformer.accuracy);

    const transformerF1 =
      Number(data.transformer.macro_f1);

    const randomForestF1 =
      Number(data.random_forest.macro_f1);


    $("transformerAccuracy").textContent =
      `${(transformerAccuracy * 100).toFixed(2)}%`;

    $("transformerF1").textContent =
      `${(transformerF1 * 100).toFixed(2)}%`;

    $("randomForestF1").textContent =
      `${(randomForestF1 * 100).toFixed(2)}%`;


    if (modelChart) modelChart.destroy();

    modelChart = new Chart($("modelComparisonChart"), {
      type: "bar",

      data: {
        labels: ["Transformer", "Random Forest"],

        datasets: [{
          label: "Macro F1 (%)",

          data: [
            transformerF1 * 100,
            randomForestF1 * 100
          ],

          backgroundColor: ["#6D8CFF", "#22C55E"],
          borderRadius: 8,
          borderWidth: 0
        }]
      },

      options: {
        responsive: true,
        maintainAspectRatio: false,

        scales: {
          y: {
            beginAtZero: true,
            max: 100,

            ticks: {
              color: "#dce6f7",

              callback: value => `${value}%`
            },

            grid: {
              color: "rgba(255,255,255,0.08)"
            }
          },

          x: {
            ticks: {
              color: "#dce6f7"
            },

            grid: {
              display: false
            }
          }
        },

        plugins: {
          legend: {
            labels: {
              color: "#dce6f7"
            }
          },

          tooltip: {
            callbacks: {
              label: context =>
                ` Macro F1: ${context.parsed.y.toFixed(2)}%`
            }
          }
        }
      }
    });

    message.textContent =
      `${data.dataset} · ${data.note}`;

  } catch (error) {
    message.textContent =
      `Could not load model performance: ${error.message}`;
  }
}


// =====================================
// REFRESH DASHBOARD
// =====================================

$("refreshBtn").addEventListener("click", () => {
  loadDashboard().catch(showError);
});


// =====================================
// REFRESH PREDICTION HISTORY
// =====================================

$("predictionHistoryRefreshBtn").addEventListener("click", () => {
  loadPredictionHistory();
});


// =====================================
// SHOW ERRORS
// =====================================

function showError(error) {
  $("formMessage").textContent =
    `Could not connect to backend: ${error.message}`;
}


// =====================================
// INITIALIZE APPLICATION
// =====================================

async function init() {
  try {
    await loadTypes();

    await loadDashboard();

    await loadPredictionHistory();

    await loadModelPerformance();

    await loadDailyInsights();

  } catch (error) {
    showError(error);
  }
}

init();
