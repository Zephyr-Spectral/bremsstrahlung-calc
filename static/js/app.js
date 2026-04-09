/**
 * Application entry point — tab management, event wiring, state.
 */
const App = (function () {
    'use strict';

    async function init() {
        await Controls.init();
        _setupTabs();
        _setupEvents();
        Controls.setStatus('Ready');
    }

    function _setupTabs() {
        document.querySelectorAll('.tab').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
                // Resize plots when tab becomes visible
                setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
            });
        });
    }

    function _setupEvents() {
        document.getElementById('calculate-btn').addEventListener('click', _calculate);
        document.getElementById('angular-calc-btn').addEventListener('click', _calcAngular);
        document.getElementById('compare-calc-btn').addEventListener('click', _calcCompare);
        document.getElementById('val-calc-btn').addEventListener('click', _calcValidation);
    }

    async function _calculate() {
        const p = Controls.getParams();
        Controls.setStatus('Calculating...');

        try {
            // Spectrum
            const qSpec = new URLSearchParams({
                material: p.material,
                electron_energy_mev: p.electron_energy_mev,
                angle_deg: p.angle_deg,
                beam_current_ua: p.beam_current_ua,
                mode: p.mode,
                n_points: 30,
            });
            const specData = await Utils.fetchJSON('/api/spectrum/calculate?' + qSpec);
            Plots.spectrum(specData);

            // Integrated
            const qInt = new URLSearchParams({
                material: p.material,
                electron_energy_mev: p.electron_energy_mev,
                n_points: 20,
            });
            const intData = await Utils.fetchJSON('/api/spectrum/integrated?' + qInt);
            Plots.integrated(intData);

            // Materials: stopping power
            const qMat = new URLSearchParams({ n_points: 30 });
            const matData = await Utils.fetchJSON(`/api/materials/${p.material}/stopping-power?` + qMat);
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
            Plots.validation(data);
            Controls.setStatus('Done');
        } catch (err) {
            Controls.setStatus('Error: ' + err.message);
        }
    }

    // Auto-init
    document.addEventListener('DOMContentLoaded', init);

    return { init };
})();
