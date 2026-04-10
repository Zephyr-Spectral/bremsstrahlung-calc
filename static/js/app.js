/**
 * Application entry point — tab management, event wiring, state.
 */
const App = (function () {
    'use strict';

    // Last-fetched data per plot, so we can re-render on theme change
    const _cache = {};

    async function init() {
        await Controls.init();
        _setupThemeToggle();
        _setupTabs();
        _setupEvents();
        Controls.setStatus('Ready');
    }

    // ── Theme ──────────────────────────────────────────────────────────────
    function _setupThemeToggle() {
        const btn = document.getElementById('theme-toggle');
        const saved = localStorage.getItem('brems-theme');
        if (saved === 'light') _applyTheme('light');

        btn.addEventListener('click', () => {
            const next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
            _applyTheme(next);
        });
    }

    function _applyTheme(theme) {
        const btn = document.getElementById('theme-toggle');
        if (theme === 'light') {
            document.documentElement.setAttribute('data-theme', 'light');
            btn.textContent = '\u2600 Dark';
        } else {
            document.documentElement.removeAttribute('data-theme');
            btn.textContent = '\u2600\ufe0e Light';
        }
        localStorage.setItem('brems-theme', theme);
        _replotAll();
    }

    function _replotAll() {
        if (_cache.spectrum) Plots.spectrum(_cache.spectrum);
        if (_cache.angular) Plots.angular(_cache.angular);
        if (_cache.integrated) Plots.integrated(_cache.integrated);
        if (_cache.compare) Plots.compare(_cache.compare);
        if (_cache.heatmap) Plots.heatmap(_cache.heatmap);
        if (_cache.validation) Plots.validation(_cache.validation);
        if (_cache.materials) Plots.materials(_cache.materials);
    }

    // ── Tabs ───────────────────────────────────────────────────────────────
    function _setupTabs() {
        document.querySelectorAll('.tab').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
                setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
            });
        });
    }

    // ── Events ─────────────────────────────────────────────────────────────
    function _setupEvents() {
        document.getElementById('calculate-btn').addEventListener('click', _calculate);
        document.getElementById('angular-calc-btn').addEventListener('click', _calcAngular);
        document.getElementById('compare-calc-btn').addEventListener('click', _calcCompare);
        document.getElementById('val-calc-btn').addEventListener('click', _calcValidation);
    }

    // ── Calculations ───────────────────────────────────────────────────────
    async function _calculate() {
        const p = Controls.getParams();
        Controls.setStatus('Calculating...');

        try {
            const qSpec = new URLSearchParams({
                material: p.material,
                electron_energy_mev: p.electron_energy_mev,
                angle_deg: p.angle_deg,
                beam_current_ua: p.beam_current_ua,
                mode: p.mode,
                n_points: 30,
            });
            const specData = await Utils.fetchJSON('/api/spectrum/calculate?' + qSpec);
            _cache.spectrum = specData;
            Plots.spectrum(specData);

            const qInt = new URLSearchParams({
                material: p.material,
                electron_energy_mev: p.electron_energy_mev,
                n_points: 20,
            });
            const intData = await Utils.fetchJSON('/api/spectrum/integrated?' + qInt);
            _cache.integrated = intData;
            Plots.integrated(intData);

            const qHeat = new URLSearchParams({
                material: p.material,
                electron_energy_mev: p.electron_energy_mev,
                n_points: 20,
                n_angles: 9,
            });
            const heatData = await Utils.fetchJSON('/api/spectrum/heatmap?' + qHeat);
            _cache.heatmap = heatData;
            Plots.heatmap(heatData);

            const qMat = new URLSearchParams({ n_points: 30 });
            const matData = await Utils.fetchJSON(`/api/materials/${p.material}/stopping-power?` + qMat);
            _cache.materials = matData;
            Plots.materials(matData);

            Controls.setStatus('Done');
        } catch (err) {
            Controls.setStatus('Error: ' + err.message);
        }
    }

    async function _calcAngular() {
        const p = Controls.getParams();
        const photonE = document.getElementById('angular-photon-energy').value;
        Controls.setStatus('Calculating angular...');

        try {
            const q = new URLSearchParams({
                material: p.material,
                electron_energy_mev: p.electron_energy_mev,
                photon_energy_mev: photonE,
                n_angles: 19,
            });
            const data = await Utils.fetchJSON('/api/spectrum/angular?' + q);
            _cache.angular = data;
            Plots.angular(data);
            Controls.setStatus('Done');
        } catch (err) {
            Controls.setStatus('Error: ' + err.message);
        }
    }

    async function _calcCompare() {
        const p = Controls.getParams();
        const mats = Controls.getCompareSelection();
        if (mats.length < 2) {
            Controls.setStatus('Select at least 2 materials');
            return;
        }
        Controls.setStatus('Comparing...');

        try {
            const q = new URLSearchParams({
                materials: mats.join(','),
                electron_energy_mev: p.electron_energy_mev,
                angle_deg: p.angle_deg,
                n_points: 20,
            });
            const data = await Utils.fetchJSON('/api/spectrum/compare?' + q);
            _cache.compare = data;
            Plots.compare(data);
            Controls.setStatus('Done');
        } catch (err) {
            Controls.setStatus('Error: ' + err.message);
        }
    }

    async function _calcValidation() {
        const p = Controls.getParams();
        const energy = document.getElementById('val-energy').value;
        const angle = document.getElementById('val-angle').value;
        Controls.setStatus('Validating...');

        try {
            const q = new URLSearchParams({
                material: p.material,
                electron_energy_mev: energy,
                angle_deg: angle,
            });
            const data = await Utils.fetchJSON('/api/validation/nasa-comparison?' + q);
            _cache.validation = data;
            Plots.validation(data);
            Controls.setStatus('Done');
        } catch (err) {
            Controls.setStatus('Error: ' + err.message);
        }
    }

    document.addEventListener('DOMContentLoaded', init);

    return { init };
})();
