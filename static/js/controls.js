/**
 * Parameter controls management.
 */
const Controls = (function () {
    'use strict';

    let _materials = [];

    async function init() {
        const data = await Utils.fetchJSON('/api/materials');
        _materials = data.materials;

        const sel = document.getElementById('material-select');
        // NASA materials first
        const nasa = _materials.filter(m => m.is_nasa);
        const ext = _materials.filter(m => !m.is_nasa);

        if (nasa.length) {
            const g1 = document.createElement('optgroup');
            g1.label = 'NASA TN D-4755';
            nasa.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.symbol;
                opt.textContent = `${m.symbol} (${m.name}, Z=${m.Z})`;
                g1.appendChild(opt);
            });
            sel.appendChild(g1);
        }
        if (ext.length) {
            const g2 = document.createElement('optgroup');
            g2.label = 'Extended';
            ext.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.symbol;
                opt.textContent = `${m.symbol} (${m.name})`;
                g2.appendChild(opt);
            });
            sel.appendChild(g2);
        }

        // Default to Cu
        sel.value = 'Cu';

        // Sync sliders and inputs
        _syncSliderInput('energy-slider', 'energy-input');
        _syncSliderInput('angle-slider', 'angle-input');

        // Build compare checkboxes
        const checksContainer = document.getElementById('compare-checks');
        _materials.forEach(m => {
            const lbl = document.createElement('label');
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.value = m.symbol;
            if (['Al', 'Cu', 'W'].includes(m.symbol)) cb.checked = true;
            lbl.appendChild(cb);
            lbl.appendChild(document.createTextNode(' ' + m.symbol));
            checksContainer.appendChild(lbl);
        });
    }

    function _syncSliderInput(sliderId, inputId) {
        const slider = document.getElementById(sliderId);
        const input = document.getElementById(inputId);
        slider.addEventListener('input', () => { input.value = slider.value; });
        input.addEventListener('change', () => { slider.value = input.value; });
    }

    function getParams() {
        const currentVal = parseFloat(document.getElementById('current-input').value) || 0;
        const unit = document.getElementById('current-unit').value;
        const beamUA = unit === 'ma' ? currentVal * 1000 : currentVal;

        return {
            material: document.getElementById('material-select').value,
            electron_energy_mev: parseFloat(document.getElementById('energy-input').value),
            angle_deg: parseFloat(document.getElementById('angle-input').value),
            beam_current_ua: beamUA,
            mode: _getTraceMode(),
        };
    }

    function _getTraceMode() {
        const checks = document.querySelectorAll('input[name="trace"]:checked');
        const vals = Array.from(checks).map(cb => cb.value);
        if (vals.length === 0) return 'calculated';
        // If all three are checked, use "all"
        if (vals.includes('calculated') && vals.includes('interpolated') && vals.includes('geant4')) return 'all';
        if (vals.includes('calculated') && vals.includes('interpolated')) return 'both';
        if (vals.includes('geant4') && vals.includes('calculated')) return 'all';
        if (vals.length === 1) return vals[0];
        return 'all';
    }

    function getSelectedTraces() {
        const checks = document.querySelectorAll('input[name="trace"]:checked');
        return Array.from(checks).map(cb => cb.value);
    }

    function getMaterials() { return _materials; }

    function getCompareSelection() {
        const checks = document.querySelectorAll('#compare-checks input:checked');
        return Array.from(checks).map(cb => cb.value);
    }

    function setStatus(msg) {
        document.getElementById('status').textContent = msg;
    }

    return { init, getParams, getMaterials, getCompareSelection, getSelectedTraces, setStatus };
})();
