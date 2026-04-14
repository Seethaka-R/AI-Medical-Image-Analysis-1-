/**
 * charts.js — Reusable Chart.js helper utilities
 */

const COLORS = {
  blue:   "#2563EB",
  green:  "#059669",
  amber:  "#D97706",
  purple: "#7C3AED",
  red:    "#DC2626",
};

/* ── Convert HEX → RGBA ───────────────────────────── */
function hexToRgba(hex, alpha = 0.07) {
  const bigint = parseInt(hex.replace("#", ""), 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/* ── Create Line Chart ────────────────────────────── */
function makeLineChart(canvasId, label1, label2, color1, color2) {
  const ctx = document.getElementById(canvasId);

  return new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: label1,
          data: [],
          borderColor: color1,
          backgroundColor: hexToRgba(color1),
          tension: 0.4,
          pointRadius: 2,
          fill: true,
        },
        {
          label: label2,
          data: [],
          borderColor: color2,
          backgroundColor: hexToRgba(color2),
          tension: 0.4,
          pointRadius: 2,
          fill: true,
        },
      ],
    },
    options: {
      animation: { duration: 250 },
      responsive: true,
      maintainAspectRatio: false,

      plugins: {
        legend: {
          position: "top",
          labels: {
            font: { size: 10 },
            usePointStyle: true,
          },
        },
      },

      scales: {
        x: {
          grid: { color: "#EFF6FF" },
          ticks: { font: { size: 9 } },
        },
        y: {
          grid: { color: "#EFF6FF" },
          ticks: { font: { size: 9 } },
          min: 0,
          max: 100,
        },
      },
    },
  });
}