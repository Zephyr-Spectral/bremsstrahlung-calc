/**
 * Shared utilities for the Bremsstrahlung Calculator.
 */
const Utils = (function () {
    'use strict';

    async function fetchJSON(url) {
        const resp = await fetch(url);
        if (!resp.ok) {
            const detail = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(detail.detail || resp.statusText);
        }
        return resp.json();
    }

    function debounce(fn, ms) {
        let timer;
        return function (...args) {
            clearTimeout(timer);
            timer = setTimeout(() => fn.apply(this, args), ms);
        };
    }

    function formatEnergy(mev) {
        if (mev >= 1) return mev.toFixed(2) + ' MeV';
        return (mev * 1000).toFixed(1) + ' keV';
    }

    function formatSci(val) {
        if (val === 0) return '0';
        return val.toExponential(3);
    }

    const _DARK = {
        paper_bgcolor: '#1a1a2e',
        plot_bgcolor: '#16213e',
        font: { color: '#e0e0e0', family: 'SF Mono, Fira Code, Consolas, monospace', size: 11 },
        xaxis: { gridcolor: '#333', zerolinecolor: '#444' },
        yaxis: { gridcolor: '#333', zerolinecolor: '#444' },
        margin: { l: 60, r: 20, t: 40, b: 50 },
        legend: { bgcolor: 'rgba(22,33,62,0.8)', bordercolor: '#333', borderwidth: 1 },
    };

    const _LIGHT = {
        paper_bgcolor: '#f4f6fb',
        plot_bgcolor: '#ffffff',
        font: { color: '#1a1a2e', family: 'SF Mono, Fira Code, Consolas, monospace', size: 11 },
        xaxis: { gridcolor: '#ccd0d9', zerolinecolor: '#aaa' },
        yaxis: { gridcolor: '#ccd0d9', zerolinecolor: '#aaa' },
        margin: { l: 60, r: 20, t: 40, b: 50 },
        legend: { bgcolor: 'rgba(255,255,255,0.9)', bordercolor: '#ccd0d9', borderwidth: 1 },
    };

    const _DARK_POLAR_BG = '#16213e';
    const _LIGHT_POLAR_BG = '#f0f4fa';

    function isLight() {
        return document.documentElement.getAttribute('data-theme') === 'light';
    }

    /** Return the active Plotly layout base object. */
    function getLayout() {
        return isLight() ? _LIGHT : _DARK;
    }

    /** Return the active polar background colour. */
    function getPolarBg() {
        return isLight() ? _LIGHT_POLAR_BG : _DARK_POLAR_BG;
    }

    const PLOT_COLORS = [
        '#e94560', '#00d2ff', '#ffd700', '#00ff88',
        '#ff6b6b', '#9b59b6', '#f39c12', '#1abc9c',
    ];

    return { fetchJSON, debounce, formatEnergy, formatSci, getLayout, getPolarBg, PLOT_COLORS };
})();
