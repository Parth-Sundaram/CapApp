import { useState, useCallback } from 'react';
import './index.css';

// ==================== RNG ====================
function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class SeededRandom {
  constructor(seed) {
    this.rand = mulberry32(seed ?? 42);
  }
  random() { return this.rand(); }
  shuffle(array) {
    const arr = [...array];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
  sample(array, n) { return this.shuffle(array).slice(0, n); }
}

// ==================== HUNGARIAN (RECTANGULAR-SAFE) ====================
function linearSumAssignment(costMatrix) {
  const n = costMatrix.length;
  const m = costMatrix[0]?.length || 0;
  const size = Math.max(n, m);

  const padded = Array.from({ length: size }, (_, i) =>
    Array.from({ length: size }, (_, j) => {
      if (i < n && j < m) return Number(costMatrix[i][j]);
      return 0;
    })
  );

  const u = Array(size + 1).fill(0);
  const v = Array(size + 1).fill(0);
  const p = Array(size + 1).fill(0);
  const way = Array(size + 1).fill(0);

  for (let i = 1; i <= size; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = Array(size + 1).fill(Infinity);
    const used = Array(size + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= size; j++) {
        if (!used[j]) {
          const cur = padded[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
      }
      for (let j = 0; j <= size; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else { minv[j] -= delta; }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  const rowInd = [], colInd = [];
  for (let j = 1; j <= size; j++) {
    const r = p[j] - 1, c = j - 1;
    if (r < n && c < m) { rowInd.push(r); colInd.push(c); }
  }
  return [rowInd, colInd];
}

// ==================== SIMULATION ====================
function simulate(myRanking, oppData, teamNames, myTeamName) {
  const matrix = teamNames.map((team) => {
    if (team === myTeamName) return [...myRanking];
    if (!oppData[team]) return null;
    return [...oppData[team]];
  });
  if (matrix.some((r) => r === null)) return -1;

  const cost = matrix.map((row) => row.map((v) => Number(v)));
  const [rowInd, colInd] = linearSumAssignment(cost);
  const assignment = {};
  rowInd.forEach((r, idx) => { assignment[r] = colInd[idx]; });
  const myRow = teamNames.indexOf(myTeamName);
  return assignment[myRow];
}

function utility(projIdx, truePref, aiSet) {
  // Base score is your exact list ranking (0 is best, 1 is second best, etc.)
  const base = truePref.indexOf(projIdx);
  // Subtract a tiny fraction for AI projects.
  // It gives a micro-bonus to AI, but NEVER enough to jump a full rank.
  return aiSet.has(projIdx) ? base - 0.1 : base;
}

function makeHonest(truePref) {
  const r = Array(truePref.length).fill(0);
  truePref.forEach((idx, rank) => { r[idx] = rank + 1; });
  return r;
}

function hillClimb(oppData, myTruePref, aiSet, teamNames, myTeamName, maxIter = 300, restarts = 5, missingTeams = [], seed = null) {
  const n = myTruePref.length;
  const rngLocal = new SeededRandom(seed ?? 42);
  const filled = { ...oppData };
  missingTeams.forEach((team) => {
    filled[team] = rngLocal.shuffle(Array.from({ length: n }, (_, i) => i + 1));
  });

  const honest = makeHonest(myTruePref);
  let bestScore = utility(simulate(honest, filled, teamNames, myTeamName), myTruePref, aiSet);
  let bestRanking = [...honest];

  for (let restart = 0; restart < restarts; restart++) {
    let current = restart === 0 ? [...honest] : rngLocal.shuffle(Array.from({ length: n }, (_, i) => i + 1));
    let currentScore = utility(simulate(current, filled, teamNames, myTeamName), myTruePref, aiSet);

    for (let it = 0; it < maxIter; it++) {
      let bestSwap = null;
      let bestSwapScore = currentScore;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          [current[i], current[j]] = [current[j], current[i]];
          const s = utility(simulate(current, filled, teamNames, myTeamName), myTruePref, aiSet);
          if (s < bestSwapScore) { bestSwapScore = s; bestSwap = [i, j]; }
          [current[i], current[j]] = [current[j], current[i]];
        }
      }
      if (!bestSwap) break;
      const [i, j] = bestSwap;
      [current[i], current[j]] = [current[j], current[i]];
      currentScore = bestSwapScore;
      if (currentScore === 0) break;
    }
    if (currentScore < bestScore) { bestScore = currentScore; bestRanking = [...current]; }
    if (bestScore === 0) break;
  }
  return [bestRanking, bestScore];
}

function injectNoise(oppData, numNoisy, numSwaps, rng, teamNames, myTeamName, missingTeams) {
  const otherTeams = teamNames.filter((t) => t !== myTeamName);
  const visibleTeams = otherTeams.filter((t) => !missingTeams.includes(t));
  const noisy = {};
  otherTeams.forEach((team) => {
    if (missingTeams.includes(team)) {
      noisy[team] = rng.shuffle(Array.from({ length: Object.values(oppData)[0].length }, (_, i) => i + 1));
    } else {
      noisy[team] = [...oppData[team]];
    }
  });
  const targets = rng.sample(visibleTeams, Math.min(numNoisy, visibleTeams.length));
  targets.forEach((team) => {
    const r = noisy[team];
    for (let s = 0; s < numSwaps; s++) {
      const [i, j] = rng.sample(Array.from({ length: r.length }, (_, idx) => idx), 2);
      [r[i], r[j]] = [r[j], r[i]];
    }
    noisy[team] = r;
  });
  return noisy;
}

function robustnessCheck({
  myTruePreferences, publicRankings, aiProjectIndices, teamNames, projectNames,
  myTeamName = 'MY_TEAM', noiseTrials = 100, seed = null, skipNoise = false,
}) {
  const n = myTruePreferences.length;
  const aiSet = new Set(aiProjectIndices);
  const otherTeams = teamNames.filter((t) => t !== myTeamName);
  const missingTeams = otherTeams.filter((t) => !(t in publicRankings));
  const projNames = projectNames || Array.from({ length: n }, (_, i) => `Proj_${String(i).zfill(2)}`);

  const honestRanking = makeHonest(myTruePreferences);
  const rngClean = new SeededRandom(seed ?? 42);
  const cleanData = { ...publicRankings };
  missingTeams.forEach((team) => {
    cleanData[team] = rngClean.shuffle(Array.from({ length: n }, (_, i) => i + 1));
  });

  const hProjClean = simulate(honestRanking, cleanData, teamNames, myTeamName);
  const hScoreClean = utility(hProjClean, myTruePreferences, aiSet);
  const [hcRanking, hcScoreClean] = hillClimb(publicRankings, myTruePreferences, aiSet, teamNames, myTeamName, 300, 5, missingTeams, seed);
  const hcProjClean = simulate(hcRanking, cleanData, teamNames, myTeamName);

  if (hcScoreClean >= hScoreClean) {
    return {
      verdict: 'SAFE', submit: 'honest', honestRanking, hcRanking,
      reason: 'HC found no improvement over honest. Nothing to risk.',
      missingTeams, projectNames: projNames, noiseResults: null,
      hProjClean, hcProjClean, noiseTrials, aiSet,
    };
  }

  if (skipNoise) {
    return {
      verdict: 'SAFE', submit: 'hc', honestRanking, hcRanking,
      reason: 'Demo mode: HC improves on clean data. Skipped noise trials.',
      missingTeams, projectNames: projNames, noiseResults: null,
      hProjClean, hcProjClean, noiseTrials, aiSet,
    };
  }

  const configs = [
    [1, 2, 'Light   (1 team, 2 swaps)'],
    [2, 4, 'Medium  (2 teams, 4 swaps)'],
    [3, 4, 'Medium  (3 teams, 4 swaps)'],
    [5, 4, 'Heavy   (5 teams, 4 swaps)'],
    [5, 8, 'Heavy   (5 teams, 8 swaps)'],
  ];
  const rng = new SeededRandom(seed ?? 42);
  const rows = [];

  for (const [numNoisy, numSwaps, label] of configs) {
    let hAi = 0, hcAi = 0, hcWins = 0, hWins = 0;
    const hPrefs = [], hcPrefs = [];
    for (let t = 0; t < noiseTrials; t++) {
      const noisy = injectNoise(publicRankings, numNoisy, numSwaps, rng, teamNames, myTeamName, missingTeams);
      const hp = simulate(honestRanking, noisy, teamNames, myTeamName);
      const hu = utility(hp, myTruePreferences, aiSet);
      hPrefs.push(hu < 1000 ? hu : 99);
      if (aiSet.has(hp)) hAi++;
      const hcp = simulate(hcRanking, noisy, teamNames, myTeamName);
      const hcu = utility(hcp, myTruePreferences, aiSet);
      hcPrefs.push(hcu < 1000 ? hcu : 99);
      if (aiSet.has(hcp)) hcAi++;
      if (hu < hcu) hWins++;
      else if (hcu < hu) hcWins++;
    }
    rows.push({
      label, hAiPct: (hAi / noiseTrials) * 100, hcAiPct: (hcAi / noiseTrials) * 100,
      hMean: hPrefs.reduce((a, b) => a + b, 0) / hPrefs.length,
      hcMean: hcPrefs.reduce((a, b) => a + b, 0) / hcPrefs.length,
      hcWins, hWins,
      hcCatPct: ((noiseTrials - hcAi) / noiseTrials) * 100,
      hCatPct: ((noiseTrials - hAi) / noiseTrials) * 100,
    });
  }

  const realistic = rows[2];
  const heavy = rows[4];
  const hcCatRealistic = realistic.hcCatPct;
  const hCatRealistic = realistic.hCatPct;
  const hcWinsRealistic = realistic.hcWins;

  let verdict, submit, reason;

  if (missingTeams.length) {
    if (hcCatRealistic > 10) {
      verdict = 'UNSAFE'; submit = 'honest';
      reason = `HC collapses under noise AND ${missingTeams.length} teams missing.`;
    } else if (hcCatRealistic > hCatRealistic + 5) {
      verdict = 'UNSAFE'; submit = 'honest';
      reason = `HC catastrophe rate (${hcCatRealistic.toFixed(0)}%) exceeds honest (${hCatRealistic.toFixed(0)}%) with missing teams.`;
    } else if (hcWinsRealistic < noiseTrials * 0.4) {
      verdict = 'RISKY'; submit = 'honest';
      reason = `HC wins only ${hcWinsRealistic}/${noiseTrials} under realistic noise. Conservative due to missing teams.`;
    } else {
      verdict = 'RISKY'; submit = 'honest';
      reason = `HC shows improvement but ${missingTeams.length} teams missing makes optimization unreliable.`;
    }
  } else {
    if (hcCatRealistic <= hCatRealistic + 5 && hcWinsRealistic >= noiseTrials * 0.5) {
      verdict = 'SAFE'; submit = 'hc';
      reason = `HC improvement survives noise. Realistic noise: HC AI=${realistic.hcAiPct.toFixed(0)}% vs Honest AI=${realistic.hAiPct.toFixed(0)}%, HC wins ${hcWinsRealistic}/${noiseTrials}.`;
    } else if (hcCatRealistic <= hCatRealistic + 15) {
      verdict = 'RISKY'; submit = 'honest';
      reason = `HC improvement is fragile. Realistic noise: HC catastrophe=${hcCatRealistic.toFixed(0)}% vs Honest catastrophe=${hCatRealistic.toFixed(0)}%. Honest wins ${realistic.hWins}/${noiseTrials}.`;
    } else {
      verdict = 'UNSAFE'; submit = 'honest';
      reason = `HC ranking collapses under noise. Realistic noise: HC catastrophe=${hcCatRealistic.toFixed(0)}% vs Honest catastrophe=${hCatRealistic.toFixed(0)}%. Honest wins ${realistic.hWins}/${noiseTrials}.`;
    }
  }

  if (heavy.hWins > noiseTrials * 0.7) {
    verdict = 'UNSAFE'; submit = 'honest';
    reason = `HC collapses under heavy noise. Honest wins ${heavy.hWins}/${noiseTrials}. The improvement is a mirage from overfitting to perfect info.`;
  }

  return {
    verdict, submit, honestRanking, hcRanking, reason, missingTeams,
    projectNames: projNames, noiseResults: rows,
    hProjClean, hcProjClean, noiseTrials, aiSet,
  };
}

// ==================== DEMO DATA ====================
const DEMO_TEAM_LIST = [...Array(14)].map((_, i) => `Team_${String(i + 1).padStart(2, '0')}`).concat(['MY_TEAM']);
const DEMO_PROJECT_LIST = [...Array(15)].map((_, i) => `Proj_${String(i).padStart(2, '0')}`);
const DEMO_AI_TEAMS = ['Team_01', 'Team_03', 'Team_05', 'Team_07', 'Team_09', 'Team_11', 'MY_TEAM'];
const DEMO_AI_INDICES = [0, 1, 2];
const DEMO_TRUE_PREF = [0, 1, 2, 5, 7, 6, 12, 13, 14, 3, 4, 8, 9, 10, 11];

function generateBloodbathRanking(n, targets, rng) {
  const desired = rng.shuffle([...targets]);
  const remaining = [];
  for (let i = 0; i < n; i++) if (!targets.includes(i)) remaining.push(i);
  const remShuffled = rng.shuffle(remaining);
  const ranking = Array(n).fill(0);
  [...desired, ...remShuffled].forEach((proj, idx) => { ranking[proj] = idx + 1; });
  return ranking;
}

function buildBaseData(teamList, n, aiTeamList, aiProjectIndices, seed) {
  const rng = new SeededRandom(seed);
  const data = {};
  for (const team of teamList) {
    if (team === 'MY_TEAM') continue;
    if (aiTeamList.includes(team)) {
      data[team] = generateBloodbathRanking(n, aiProjectIndices, rng);
    } else {
      data[team] = rng.shuffle(Array.from({ length: n }, (_, i) => i + 1));
    }
  }
  return data;
}

function getDemoPreset(id) {
  if (id === 'baseline') {
    return { desc: 'Baseline random opponents', data: buildBaseData(DEMO_TEAM_LIST, 15, DEMO_AI_TEAMS, DEMO_AI_INDICES, 1) };
  }
  if (id === 'cluster') {
    const rng = new SeededRandom(1);
    const data = {};
    for (const team of DEMO_TEAM_LIST) {
      if (team === 'MY_TEAM') continue;
      const top = rng.shuffle([0, 1, 2, 3, 4]);
      const bot = rng.shuffle([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
      const r = Array(15).fill(0);
      [...top, ...bot].forEach((p, i) => (r[p] = i + 1));
      data[team] = r;
    }
    return { desc: 'Top-5 clustering', data };
  }
  if (id === 'partial') {
    const base = getDemoPreset('cluster').data;
    const data = {};
    Object.entries(base).forEach(([k, v]) => { if (k !== 'Team_06' && k !== 'Team_12') data[k] = v; });
    return { desc: 'Clustering with 2 missing teams', data };
  }
  if (id === 'catastrophe') {
    return { desc: 'Catastrophe save scenario (seed 11)', data: buildBaseData(DEMO_TEAM_LIST, 15, DEMO_AI_TEAMS, DEMO_AI_INDICES, 11) };
  }
  return { desc: '', data: {} };
}

// ==================== UI ====================
export default function App() {
  const myTeamName = 'MY_TEAM';

  const [projectNames, setProjectNames] = useState([...DEMO_PROJECT_LIST]);
  const [teamNames, setTeamNames] = useState([...DEMO_TEAM_LIST]);
  const [myTruePref, setMyTruePref] = useState([...DEMO_TRUE_PREF]);
  const [aiProjectIndices, setAiProjectIndices] = useState(new Set(DEMO_AI_INDICES));
  const [publicRankings, setPublicRankings] = useState(() => getDemoPreset('baseline').data);
  const [jsonInput, setJsonInput] = useState(() => JSON.stringify(getDemoPreset('baseline').data, null, 2));
  const [noiseTrials, setNoiseTrials] = useState(100);
  const [seed, setSeed] = useState(42);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [demoMode, setDemoMode] = useState(true);
  const [dragOverIndex, setDragOverIndex] = useState(null);
  const [jsonStatus, setJsonStatus] = useState({ kind: 'ok', message: 'valid · 14 teams' });
  const [templateCount, setTemplateCount] = useState(14);

  // Helper: parse JSON and propagate updates. Called by the JSON editor's
  // onChange and by loadPreset. This lives in event-handler territory —
  // not in a useEffect — so it doesn't trigger React's "setState in effect"
  // lint warning, and it runs exactly once per user action.
  const handleJsonInputChange = useCallback((nextJson, nextParsedOverride) => {
  setJsonInput(nextJson);

  let parsed = nextParsedOverride;
  if (parsed === undefined) {
    if (!nextJson.trim()) {
      setJsonStatus({ kind: 'idle', message: 'empty' });
      setPublicRankings({});
      return;
    }
    try {
      parsed = JSON.parse(nextJson);
    } catch (e) {
      setJsonStatus({ kind: 'err', message: e.message });
      return;
    }
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      setJsonStatus({ kind: 'err', message: 'must be an object {team: [ranks]}' });
      return;
    }
  }

    const firstRanking = Object.values(parsed)[0];
    const detectedN = firstRanking?.length || 15;

    setProjectNames((prev) => {
      if (detectedN <= prev.length) return prev;
      const extra = Array.from(
        { length: detectedN - prev.length },
        (_, i) => `Proj_${String(prev.length + i).padStart(2, '0')}`
      );
      return [...prev, ...extra];
    });

    setMyTruePref((prev) => {
      if (detectedN <= prev.length) return prev;
      const current = new Set(prev);
      const extra = [];
      for (let i = 0; i < detectedN; i++) if (!current.has(i)) extra.push(i);
      return [...prev, ...extra];
    });

    setTeamNames((prev) => {
      const jsonTeams = Object.keys(parsed);
      if (!jsonTeams.includes(myTeamName)) {
        return [...jsonTeams, myTeamName];
      }
      return jsonTeams;
    });

    setPublicRankings((prev) => {
      const prevKeys = Object.keys(prev);
      const nextKeys = Object.keys(parsed);
      if (prevKeys.length === nextKeys.length) {
        let same = true;
        for (const k of nextKeys) {
          const a = prev[k];
          const b = parsed[k];
          if (!a || a.length !== b.length) { same = false; break; }
          for (let i = 0; i < a.length; i++) {
            if (a[i] !== b[i]) { same = false; break; }
          }
          if (!same) break;
        }
        if (same) return prev;
      }
      return parsed;
    });

    const keys = Object.keys(parsed);
    const okMsg = `valid · ${keys.length} team${keys.length === 1 ? '' : 's'}`;
    setJsonStatus((prev) =>
      prev.kind === 'ok' && prev.message === okMsg ? prev : { kind: 'ok', message: okMsg }
    );
  }, []);

  const movePref = useCallback((from, to) => {
    setMyTruePref((prev) => {
      if (from === to || to < 0 || to >= prev.length) return prev;
      const next = [...prev];
      const [item] = next.splice(from, 1);
      next.splice(to, 0, item);
      return next;
    });
  }, []);

  const toggleAi = useCallback((projIdx) => {
    setAiProjectIndices((prev) => {
      const next = new Set(prev);
      if (next.has(projIdx)) next.delete(projIdx);
      else next.add(projIdx);
      return next;
    });
  }, []);

  const renameProject = useCallback((idx, newName) => {
    setProjectNames((prev) => {
      const next = [...prev];
      next[idx] = newName;
      return next;
    });
  }, []);

  const loadPreset = (id) => {
    const preset = getDemoPreset(id);
    handleJsonInputChange(JSON.stringify(preset.data, null, 2), preset.data);
  };

  const handleGenerateTemplate = () => {
    const n = projectNames.length;
    const blankRanking = Array.from({ length: n }, (_, i) => i + 1);
    
    const template = {};
    for (let i = 1; i <= templateCount; i++) {
      const teamName = `Team_${String(i).padStart(2, '0')}`;
      template[teamName] = [...blankRanking];
    }
    
    handleJsonInputChange(JSON.stringify(template, null, 2), template);
  };

  const runAnalysis = () => {
    const n = projectNames.length;
    for (const [team, r] of Object.entries(publicRankings)) {
      if (r.length !== n) {
        alert(`${team} ranking has ${r.length} entries, expected ${n}`);
        return;
      }
      const sorted = [...r].sort((a, b) => a - b);
      const valid = sorted.every((v, i) => v === i + 1);
      if (!valid) {
        alert(`${team} ranking is not a valid 1..${n} permutation`);
        return;
      }
    }

    setLoading(true);
    setTimeout(() => {
      const check = robustnessCheck({
        myTruePreferences: myTruePref,
        publicRankings,
        aiProjectIndices: Array.from(aiProjectIndices),
        teamNames,
        projectNames,
        myTeamName,
        noiseTrials,
        seed,
        skipNoise: demoMode,
      });
      setResult(check);
      setLoading(false);
    }, 50);
  };

  const copyRanking = (ranking) => {
    navigator.clipboard.writeText(JSON.stringify(ranking));
  };

  const honestProj = result ? projectNames[result.hProjClean] : null;
  const hcProj = result ? projectNames[result.hcProjClean] : null;

  // Drag handlers
  const handleDragStart = (e, index) => {
    e.dataTransfer.setData('text/plain', String(index));
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e, index) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setDragOverIndex(index);
  };

  const handleDragLeave = () => {
    setDragOverIndex(null);
  };

  const handleDrop = (e, toIndex) => {
    e.preventDefault();
    const fromIndex = Number(e.dataTransfer.getData('text/plain'));
    movePref(fromIndex, toIndex);
    setDragOverIndex(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Optimal Ranking</h1>
        <p>Assignment strategy optimizer — Hungarian algorithm + robustness stress-testing</p>
      </header>

      <section className="card">
        <h2>Your True Preferences</h2>
        <p className="label" style={{ marginTop: -8 }}>Drag to reorder · double-click a name to rename</p>
        <div className="pref-list">
          {myTruePref.map((projIdx, rank) => (
            <PrefRow
              key={projIdx}
              projIdx={projIdx}
              rank={rank}
              name={projectNames[projIdx]}
              isAi={aiProjectIndices.has(projIdx)}
              isDragOver={dragOverIndex === rank}
              onRename={(newName) => renameProject(projIdx, newName)}
              onDragStart={(e) => handleDragStart(e, rank)}
              onDragOver={(e) => handleDragOver(e, rank)}
              onDragLeave={handleDragLeave}
              onDrop={(e) => handleDrop(e, rank)}
            />
          ))}
        </div>
      </section>

      <section className="card">
        <h2>AI-Capable Projects</h2>
        <p className="label" style={{ marginTop: -8 }}>Toggle which projects require AI</p>
        <div className="ai-grid">
          {projectNames.map((name, idx) => (
            <button
              key={idx}
              className={`ai-chip ${aiProjectIndices.has(idx) ? 'active' : ''}`}
              onClick={() => toggleAi(idx)}
            >
              {name}
            </button>
          ))}
        </div>
      </section>

      <section className="card">
        <h2>Opponent Rankings</h2>
        <div className="flex gap-2 mb-4">
          <button className="btn btn-secondary btn-sm" onClick={() => loadPreset('baseline')}>Demo: Baseline</button>
          <button className="btn btn-secondary btn-sm" onClick={() => loadPreset('cluster')}>Demo: Cluster</button>
          <button className="btn btn-secondary btn-sm" onClick={() => loadPreset('partial')}>Demo: Partial</button>
          <button className="btn btn-secondary btn-sm" onClick={() => loadPreset('catastrophe')}>Demo: Catastrophe</button>
        </div>

        <div className="flex justify-between items-end mb-2 mt-4">
          <p className="label" style={{ margin: 0 }}>JSON Input</p>
          <div className="flex gap-2 items-center text-fog" style={{ fontSize: '0.85rem' }}>
            Template for: 
            <input 
              type="number" 
              className="template-input"
              value={templateCount} 
              onChange={(e) => setTemplateCount(Math.max(1, Number(e.target.value)))}
            />
            teams
            <button className="btn btn-secondary btn-sm" onClick={handleGenerateTemplate}>Generate</button>
          </div>
        </div>

        <JsonEditor
          value={jsonInput}
          onChange={handleJsonInputChange}
          minRows={6}
          placeholder={`{\n  "Team_01": [3, 1, 2, 5, 4, ...],\n  "Team_02": [2, 5, 1, ...]\n}`}
        />
        
        <div className="flex justify-between items-center mt-2">
          <span className="text-fog mono" style={{ fontSize: '0.78rem' }}>
            {Object.keys(publicRankings).length} teams · auto-parsed on edit
          </span>
          <span className={`json-status-pill ${jsonStatus.kind}`}>
            <span className="json-status-dot" />
            {jsonStatus.message}
          </span>
        </div>
      </section>

      <section className="card">
        <h2>Settings</h2>
        <div className="result-grid">
          <div>
            <p className="label">Noise Trials</p>
            <input type="number" value={noiseTrials} onChange={(e) => setNoiseTrials(Number(e.target.value))} />
          </div>
          <div>
            <p className="label">Random Seed</p>
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </div>
        </div>

        <div className="flex items-center gap-3 mt-4" style={{ padding: '12px', background: 'var(--paper)', borderRadius: 8, border: '1px solid var(--rule)' }}>
          <button
            className={`btn btn-sm ${demoMode ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setDemoMode(true)}
          >
            Demo Mode
          </button>
          <button
            className={`btn btn-sm ${!demoMode ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setDemoMode(false)}
          >
            Real Data
          </button>
          <span className="text-fog" style={{ fontSize: '0.85rem' }}>
            {demoMode ? 'Skips noise trials — instant results for testing' : 'Full robustness check with noise stress test'}
          </span>
        </div>
      </section>

      <button className="btn btn-primary w-full mt-4" onClick={runAnalysis} disabled={loading}>
        {loading ? 'Running Hungarian + Hill Climb…' : demoMode ? 'Run Quick Check' : 'Run Robustness Check'}
      </button>

      {result && (
        <div style={{ marginTop: 32 }}>
          <div className={`verdict-banner verdict-${result.verdict.toLowerCase()}`}>
            <h2>
              {result.verdict === 'SAFE' && 'SAFE — Submit HC Ranking'}
              {result.verdict === 'RISKY' && 'RISKY — Submit Honest Ranking'}
              {result.verdict === 'UNSAFE' && 'UNSAFE — Submit Honest Ranking'}
            </h2>
            <p style={{ margin: '8px 0 0', maxWidth: 720 }}>{result.reason}</p>
            {result.missingTeams.length > 0 && (
              <p className="text-fog" style={{ marginTop: 8, fontSize: '0.85rem' }}>
                Missing teams: {result.missingTeams.join(', ')}
              </p>
            )}
          </div>

          <div className="result-grid">
            <div className="result-card">
              <h4>Honest Outcome</h4>
              <div className="result-project">{honestProj}</div>
              <div className="result-meta mono">
                {result.aiSet.has(result.hProjClean) ? 'AI Project' : 'Non-AI'} · True pref #{myTruePref.indexOf(result.hProjClean) + 1}
              </div>
            </div>
            <div className="result-card">
              <h4>HC Outcome</h4>
              <div className="result-project">{hcProj}</div>
              <div className="result-meta mono">
                {result.aiSet.has(result.hcProjClean) ? 'AI Project' : 'Non-AI'} · True pref #{myTruePref.indexOf(result.hcProjClean) + 1}
              </div>
            </div>
          </div>

          <div className={`submit-block ${result.submit === 'honest' ? 'safe' : ''}`}>
            <div className="flex justify-between items-center">
              <h3 style={{ fontFamily: 'Newsreader, serif' }}>
                {result.submit === 'hc' ? 'Submit: HC Ranking' : 'Submit: Honest Ranking'}
              </h3>
              <button className="btn btn-sm btn-primary" onClick={() => copyRanking(result.submit === 'hc' ? result.hcRanking : result.honestRanking)}>Copy</button>
            </div>
            <div className="code-block mono">
              {JSON.stringify(result.submit === 'hc' ? result.hcRanking : result.honestRanking)}
            </div>
          </div>

          {result.noiseResults && (
            <div className="card mt-4">
              <h2>Noise Stress Test</h2>
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table mono">
                  <thead>
                    <tr>
                      <th>Scenario</th>
                      <th>Hon AI</th>
                      <th>HC AI</th>
                      <th>HC Wins</th>
                      <th>Hon Wins</th>
                      <th>HC Cat%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.noiseResults.map((row, i) => (
                      <tr key={i}>
                        <td style={{ fontFamily: 'Inter Tight, sans-serif' }}>{row.label}</td>
                        <td>{row.hAiPct.toFixed(0)}%</td>
                        <td>{row.hcAiPct.toFixed(0)}%</td>
                        <td className={row.hcWins > row.hWins ? 'text-olive' : ''}>{row.hcWins}/{result.noiseTrials}</td>
                        <td className={row.hWins > row.hcWins ? 'text-ember' : ''}>{row.hWins}/{result.noiseTrials}</td>
                        <td>{row.hcCatPct.toFixed(0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {result.submit === 'hc' && (
            <div className="card mt-4">
              <h2>What Changed vs Honest</h2>
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table mono">
                  <thead>
                    <tr>
                      <th style={{ fontFamily: 'Inter Tight, sans-serif' }}>Project</th>
                      <th>Honest</th>
                      <th>HC</th>
                      <th>Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {projectNames.map((name, idx) => {
                      const h = result.honestRanking[idx];
                      const hc = result.hcRanking[idx];
                      if (h === hc) return null;
                      const delta = hc - h;
                      return (
                        <tr key={idx}>
                          <td style={{ fontFamily: 'Inter Tight, sans-serif' }}>
                            {name}
                            {result.aiSet.has(idx) && <span className="ai-badge" style={{ marginLeft: 8 }}>AI</span>}
                          </td>
                          <td>{h}</td>
                          <td>{hc}</td>
                          <td className={delta > 0 ? 'text-ember' : 'text-olive'}>
                            {delta > 0 ? `DN ${delta}` : `UP ${Math.abs(delta)}`}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ==================== PREF ROW (with double-click rename) ====================
function PrefRow({ rank, name, isAi, isDragOver, onRename, onDragStart, onDragOver, onDragLeave, onDrop }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(name);

  const startEdit = () => {
    setDraft(name);
    setEditing(true);
  };

  const commit = () => {
    const trimmed = draft.trim();
    if (trimmed && trimmed !== name) onRename(trimmed);
    setEditing(false);
  };

  const cancel = () => {
    setDraft(name);
    setEditing(false);
  };

  return (
    <div
      className={`pref-item ${isDragOver ? 'drag-over' : ''} ${editing ? 'editing' : ''}`}
      draggable={!editing}
      onDragStart={onDragStart}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <span className="pref-rank mono">{rank + 1}</span>
      {editing ? (
        <input
          type="text"
          autoFocus
          className="pref-name-edit"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === 'Enter') { e.preventDefault(); commit(); }
            else if (e.key === 'Escape') { e.preventDefault(); cancel(); }
          }}
          onClick={(e) => e.stopPropagation()}
        />
      ) : (
        <span
          className="pref-name"
          onDoubleClick={startEdit}
          title="Double-click to rename"
        >
          {name}
        </span>
      )}
      {isAi && <span className="ai-badge">AI</span>}
      <span className="drag-handle" title="Drag to reorder">⋮⋮</span>
    </div>
  );
}

// ==================== JSON EDITOR (Prettier-style highlighting) ====================
function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function highlightJson(text) {
  const safe = escapeHtml(text);
  let out = '';
  let i = 0;
  const n = safe.length;
  while (i < n) {
    const c = safe[i];
    if (c === '"') {
      let j = i + 1;
      while (j < n) {
        if (safe[j] === '\\' && j + 1 < n) { j += 2; continue; }
        if (safe[j] === '"') break;
        j++;
      }
      const str = safe.slice(i, j + 1);
      let k = j + 1;
      while (k < n && (safe[k] === ' ' || safe[k] === '\t')) k++;
      const isKey = safe[k] === ':';
      out += `<span class="${isKey ? 'tok-key' : 'tok-string'}">${str}</span>`;
      i = j + 1;
    } else if (
      (c === '-' || (c >= '0' && c <= '9')) &&
      (i === 0 || /[\s,[{:]/.test(safe[i - 1]))
    ) {
      let j = i;
      if (safe[j] === '-') j++;
      while (j < n && /[0-9.eE+-]/.test(safe[j])) j++;
      out += `<span class="tok-number">${safe.slice(i, j)}</span>`;
      i = j;
    } else if (c === '{' || c === '}' || c === '[' || c === ']') {
      out += `<span class="tok-bracket">${c}</span>`;
      i++;
    } else if (c === ',' || c === ':') {
      out += `<span class="tok-punct">${c}</span>`;
      i++;
    } else if (c === 't' && safe.slice(i, i + 4) === 'true') {
      out += `<span class="tok-literal">true</span>`;
      i += 4;
    } else if (c === 'f' && safe.slice(i, i + 5) === 'false') {
      out += `<span class="tok-literal">false</span>`;
      i += 5;
    } else if (c === 'n' && safe.slice(i, i + 4) === 'null') {
      out += `<span class="tok-literal">null</span>`;
      i += 4;
    } else {
      out += c;
      i++;
    }
  }
  return out;
}

function JsonEditor({ value, onChange, placeholder, minRows = 6 }) {
  const lineCount = Math.max(value.split('\n').length, minRows);
  const lineNumbers = Array.from({ length: lineCount }, (_, i) => i + 1).join('\n');
  const highlighted = highlightJson(value || '');

  const handleScroll = (e) => {
    const overlay = e.target.parentElement.querySelector('.json-highlight');
    const gutter = e.target.parentElement.parentElement.querySelector('.json-gutter');
    if (overlay) {
      overlay.scrollTop = e.target.scrollTop;
      overlay.scrollLeft = e.target.scrollLeft;
    }
    if (gutter) {
      gutter.scrollTop = e.target.scrollTop;
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const ta = e.target;
      const start = ta.selectionStart;
      const end = ta.selectionEnd;
      const newValue = value.slice(0, start) + '  ' + value.slice(end);
      onChange(newValue);
      requestAnimationFrame(() => {
        ta.selectionStart = ta.selectionEnd = start + 2;
      });
    }
  };

  return (
    <div className="json-editor">
      <div className="json-editor-body">
        <pre className="json-gutter">{lineNumbers}</pre>
        <div className="json-editor-content">
          <pre
            className="json-highlight"
            dangerouslySetInnerHTML={{ __html: highlighted + '\n' }}
            aria-hidden="true"
          />
          <textarea
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onScroll={handleScroll}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            spellCheck={false}
            autoCorrect="off"
            autoCapitalize="off"
            rows={minRows}
          />
        </div>
      </div>
    </div>
  );
}
