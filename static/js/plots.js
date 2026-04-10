/**
 * Plotly.js visualization functions.
 */
const Plots = (function () {
    'use strict';

    const C = Utils.PLOT_COLORS;

    function spectrum(data) {
        const L = Utils.getLayout();
        const traces = [];
        const params = data.parameters;

        if (data.calculated) {
            traces.push({
                x: data.calculated.photon_energy_mev,
                y: data.calculated.intensity,
                name: 'Calculated',
                line: { color: C[0], width: 2 },
                type: 'scatter',
            });
        }
        if (data.interpolated) {
            traces.push({
                x: data.interpolated.photon_energy_mev,
                y: data.interpolated.intensity,
                name: 'NASA Interpolated',
                mode: 'lines+markers',
                line: { color: C[1], width: 2, dash: 'dash' },
                marker: { size: 4 },
                type: 'scatter',
            });
        }
        if (data.geant4) {
            traces.push({
                x: data.geant4.photon_energy_mev,
                y: data.geant4.intensity,
                name: `Geant4 MC (${(data.geant4.n_events/1000).toFixed(0)}K)`,
                mode: 'lines+markers',
                line: { color: '#50fa7b', width: 2 },
                marker: { size: 5, symbol: 'diamond' },
                type: 'scatter',
            });
        }

        const layout = {
            ...L,
            title: `Bremsstrahlung Spectrum: ${params.material}, E=${params.electron_energy_mev} MeV, ${params.angle_deg}\u00b0`,
            xaxis: { ...L.xaxis, title: 'Photon Energy (MeV)', type: 'log' },
            yaxis: { ...L.yaxis, title: 'Intensity (MeV / MeV\u00b7sr\u00b7e\u207b)', type: 'log' },
        };

        Plotly.react('plot-spectrum', traces, layout, { responsive: true });
    }

    function angular(data) {
        const L = Utils.getLayout();
        const trace = {
            theta: data.angles_deg,
            r: data.intensity,
            name: data.parameters.material,
            type: 'scatterpolar',
            mode: 'lines+markers',
            line: { color: C[0], width: 2 },
            marker: { size: 4 },
        };

        const gridColor = Utils.getPolarBg() === '#16213e' ? '#333' : '#ccd0d9';
        const layout = {
            ...L,
            title: `Angular Distribution: ${data.parameters.material}, E\u2090=${data.parameters.electron_energy_mev} MeV, k=${data.parameters.photon_energy_mev} MeV`,
            polar: {
                bgcolor: Utils.getPolarBg(),
                angularaxis: { gridcolor: gridColor, linecolor: gridColor, tickfont: { color: L.font.color } },
                radialaxis: { gridcolor: gridColor, linecolor: gridColor, tickfont: { color: L.font.color }, type: 'log' },
            },
        };

        Plotly.react('plot-angular', [trace], layout, { responsive: true });
    }

    function integrated(data) {
        const L = Utils.getLayout();
        const trace = {
            x: data.photon_energy_mev,
            y: data.intensity,
            name: 'Angle-integrated',
            line: { color: C[0], width: 2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(233,69,96,0.15)',
            type: 'scatter',
        };

        const layout = {
            ...L,
            title: `Angle-Integrated Spectrum: ${data.parameters.material}, E=${data.parameters.electron_energy_mev} MeV`,
            xaxis: { ...L.xaxis, title: 'Photon Energy (MeV)', type: 'log' },
            yaxis: { ...L.yaxis, title: 'Intensity (MeV / MeV\u00b7e\u207b)', type: 'log' },
        };

        Plotly.react('plot-integrated', [trace], layout, { responsive: true });
    }

    function compare(data) {
        const L = Utils.getLayout();
        const traces = [];
        const symbols = Object.keys(data.spectra);
        symbols.forEach((sym, i) => {
            const s = data.spectra[sym];
            traces.push({
                x: s.photon_energy_mev,
                y: s.intensity,
                name: sym,
                line: { color: C[i % C.length], width: 2 },
                type: 'scatter',
            });
        });

        const layout = {
            ...L,
            title: `Material Comparison: E=${data.parameters.electron_energy_mev} MeV, ${data.parameters.angle_deg}\u00b0`,
            xaxis: { ...L.xaxis, title: 'Photon Energy (MeV)', type: 'log' },
            yaxis: { ...L.yaxis, title: 'Intensity (MeV / MeV\u00b7sr\u00b7e\u207b)', type: 'log' },
        };

        Plotly.react('plot-compare', traces, layout, { responsive: true });
    }

    function heatmap(data) {
        const L = Utils.getLayout();
        const trace = {
            x: data.photon_energy_mev,
            y: data.angles_deg,
            z: data.intensity,
            type: 'heatmap',
            colorscale: 'Viridis',
            colorbar: {
                title: { text: 'Intensity', side: 'right' },
                tickfont: { color: L.font.color },
                titlefont: { color: L.font.color },
            },
        };

        const layout = {
            ...L,
            title: `Intensity Heatmap: ${data.parameters.material}, E=${data.parameters.electron_energy_mev} MeV`,
            xaxis: { ...L.xaxis, title: 'Photon Energy (MeV)', type: 'log' },
            yaxis: { ...L.yaxis, title: 'Angle (deg)' },
        };

        Plotly.react('plot-heatmap', [trace], layout, { responsive: true });
    }

    function validation(data) {
        const L = Utils.getLayout();
        const traces = [];

        if (data.nasa) {
            traces.push({
                x: data.nasa.photon_energy_mev,
                y: data.nasa.intensity,
                name: 'NASA Data',
                mode: 'markers',
                marker: { color: C[1], size: 8, symbol: 'circle' },
                type: 'scatter',
            });
        }
        if (data.calculated) {
            traces.push({
                x: data.calculated.photon_energy_mev,
                y: data.calculated.intensity,
                name: 'Calculated',
                line: { color: C[0], width: 2 },
                type: 'scatter',
            });
        }

        const p = data.parameters;
        const layout = {
            ...L,
            title: `Validation: ${p.material}, E=${p.electron_energy_mev} MeV, ${p.angle_deg}\u00b0`,
            xaxis: { ...L.xaxis, title: 'Photon Energy (MeV)' },
            yaxis: { ...L.yaxis, title: 'Intensity', type: 'log' },
        };

        Plotly.react('plot-validation', traces, layout, { responsive: true });
    }

    function materials(data) {
        const L = Utils.getLayout();
        const traces = [{
            x: data.electron_energy_mev,
            y: data.stopping_power_mev_cm2_g || data.range_g_cm2,
            name: data.material,
            line: { color: C[0], width: 2 },
            type: 'scatter',
        }];

        const isRange = !!data.range_g_cm2;
        const layout = {
            ...L,
            title: `${isRange ? 'Electron Range' : 'Stopping Power'}: ${data.material}`,
            xaxis: { ...L.xaxis, title: 'Electron Energy (MeV)', type: 'log' },
            yaxis: {
                ...L.yaxis,
                title: isRange ? 'Range (g/cm\u00b2)' : 'Stopping Power (MeV\u00b7cm\u00b2/g)',
                type: 'log',
            },
        };

        Plotly.react('plot-materials', traces, layout, { responsive: true });
    }

    return { spectrum, angular, integrated, compare, heatmap, validation, materials };
})();
