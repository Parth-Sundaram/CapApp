import { useState, useCallback, useEffect, useRef } from "react";
import { initializeApp, getApps } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getFirestore, doc, getDoc, setDoc } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

// ── Firebase setup ──
const firebaseConfig = {
  apiKey: "AIzaSyAHaJAk1UwctosPN7vDANkS8qCiyEur2bw",
  authDomain: "capapp-14536.firebaseapp.com",
  projectId: "capapp-14536",
  storageBucket: "capapp-14536.firebasestorage.app",
  messagingSenderId: "234766946911",
  appId: "1:234766946911:web:257a8c0bd5bcf28eed01ea",
};
const fbApp = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];
const db = getFirestore(fbApp);
const CONFIG_DOC = doc(db, "config", "main");

async function loadFromFirestore() {
  try {
    const snap = await getDoc(CONFIG_DOC);
    if (snap.exists()) return snap.data();
  } catch (e) { console.warn("Firestore load failed:", e); }
  return null;
}
async function saveToFirestore(data) {
  try { await setDoc(CONFIG_DOC, data, { merge: true }); }
  catch (e) { console.warn("Firestore save failed:", e); }
}

// ── debounce hook with patch merging ──
// Multiple calls within `delay` ms are coalesced: the final write merges all
// pending patches via Object.assign. This fixes the bug where rapid sequential
// setAndSave calls (e.g. during bulk Import) silently overwrote each other,
// resulting in only the last patch being persisted.
function useDebouncedMerge(fn, delay) {
  const timer = useRef(null);
  const pending = useRef({});
  const fnRef = useRef(fn);
  fnRef.current = fn;
  return useCallback((patch) => {
    pending.current = { ...pending.current, ...patch };
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      const toSend = pending.current;
      pending.current = {};
      fnRef.current(toSend);
    }, delay);
  }, [delay]);
}

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
  constructor(seed) { this.rand = mulberry32(seed ?? 42); }
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

// ==================== HUNGARIAN ====================
function linearSumAssignment(costMatrix) {
  const n = costMatrix.length, m = costMatrix[0]?.length || 0, size = Math.max(n, m);
  const padded = Array.from({ length: size }, (_, i) =>
    Array.from({ length: size }, (_, j) => (i < n && j < m ? Number(costMatrix[i][j]) : 0))
  );
  const u = Array(size + 1).fill(0), v = Array(size + 1).fill(0);
  const p = Array(size + 1).fill(0), way = Array(size + 1).fill(0);
  for (let i = 1; i <= size; i++) {
    p[0] = i; let j0 = 0;
    const minv = Array(size + 1).fill(Infinity), used = Array(size + 1).fill(false);
    do {
      used[j0] = true; const i0 = p[j0]; let delta = Infinity, j1 = 0;
      for (let j = 1; j <= size; j++) {
        if (!used[j]) {
          const cur = padded[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
      }
      for (let j = 0; j <= size; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const rowInd = [], colInd = [];
  for (let j = 1; j <= size; j++) { const r = p[j] - 1, c = j - 1; if (r < n && c < m) { rowInd.push(r); colInd.push(c); } }
  return [rowInd, colInd];
}

// ==================== SIMULATION ====================
function simulate(myRanking, oppData, teamNames, myTeamName) {
  const matrix = teamNames.map((team, idx) => {
    if (team === myTeamName) return [...myRanking];

    // Fallback logic: check the hardcoded name, then the watcher name
    const anonName = `Anonymous_${idx + 1}`;
    const data = oppData[team] || oppData[anonName];

    if (!data) return null;
    return [...data];
  });

  if (matrix.some((r) => r === null)) return -1;

  const [rowInd, colInd] = linearSumAssignment(matrix.map((r) => r.map(Number)));
  const assignment = {};
  rowInd.forEach((r, idx) => { assignment[r] = colInd[idx]; });
  
  return assignment[teamNames.indexOf(myTeamName)];
}
function utility(projIdx, truePref, aiSet) {
  const base = truePref.indexOf(projIdx);
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
  missingTeams.forEach((team) => { filled[team] = rngLocal.shuffle(Array.from({ length: n }, (_, i) => i + 1)); });
  const honest = makeHonest(myTruePref);
  let bestScore = utility(simulate(honest, filled, teamNames, myTeamName), myTruePref, aiSet);
  let bestRanking = [...honest];
  for (let restart = 0; restart < restarts; restart++) {
    let current = restart === 0 ? [...honest] : rngLocal.shuffle(Array.from({ length: n }, (_, i) => i + 1));
    let currentScore = utility(simulate(current, filled, teamNames, myTeamName), myTruePref, aiSet);
    for (let it = 0; it < maxIter; it++) {
      let bestSwap = null, bestSwapScore = currentScore;
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
  const n = Object.values(oppData)[0]?.length || 32;
  otherTeams.forEach((team) => {
    noisy[team] = missingTeams.includes(team) ? rng.shuffle(Array.from({ length: n }, (_, i) => i + 1)) : [...oppData[team]];
  });
  rng.sample(visibleTeams, Math.min(numNoisy, visibleTeams.length)).forEach((team) => {
    const r = noisy[team];
    for (let s = 0; s < numSwaps; s++) { const [i, j] = rng.sample(Array.from({ length: r.length }, (_, idx) => idx), 2); [r[i], r[j]] = [r[j], r[i]]; }
    noisy[team] = r;
  });
  return noisy;
}
function robustnessCheck({ 
  myTruePreferences, 
  publicRankings, 
  aiProjectIndices, 
  teamNames, 
  projectNames, 
  myTeamName = "MY_TEAM", 
  noiseTrials = 100, 
  seed = null, 
  skipNoise = false 
}) {
  const n = myTruePreferences.length;
  const aiSet = new Set(aiProjectIndices);
  const otherTeams = teamNames.filter((t) => t !== myTeamName);

  // 1. Correctly identify missing teams (neither "Team_X" nor "Anonymous_X" exists)
  const missingTeams = otherTeams.filter((t, idx) => {
    const anonName = `Anonymous_${idx + 1}`;
    return !(t in publicRankings) && !(anonName in publicRankings);
  });

  const projNames = projectNames || Array.from({ length: n }, (_, i) => `Proj_${String(i).padStart(2, "0")}`);
  const honestRanking = makeHonest(myTruePreferences);
  const rngClean = new SeededRandom(seed ?? 42);

  // 2. Prepare consolidated "cleanData" for simulation
  // This maps everyone to their "Team_X" name for the matrix math
  const cleanData = {};
  otherTeams.forEach((team, idx) => {
    const anonName = `Anonymous_${idx + 1}`;
    const existingData = publicRankings[team] || publicRankings[anonName];
    
    if (existingData) {
      cleanData[team] = [...existingData];
    } else {
      cleanData[team] = rngClean.shuffle(Array.from({ length: n }, (_, i) => i + 1));
    }
  });

  const hProjClean = simulate(honestRanking, cleanData, teamNames, myTeamName);
  const hScoreClean = utility(hProjClean, myTruePreferences, aiSet);

  // 3. Hill Climb based on the consolidated clean data
  const [hcRanking, hcScoreClean] = hillClimb(cleanData, myTruePreferences, aiSet, teamNames, myTeamName, 300, 5, missingTeams, seed);
  const hcProjClean = simulate(hcRanking, cleanData, teamNames, myTeamName);

  if (hcScoreClean >= hScoreClean) {
    return { verdict: "SAFE", submit: "honest", honestRanking, hcRanking, reason: "HC found no improvement over honest. Nothing to risk.", missingTeams, projectNames: projNames, noiseResults: null, hProjClean, hcProjClean, noiseTrials, aiSet };
  }
  
  if (skipNoise) {
    return { verdict: "SAFE", submit: "hc", honestRanking, hcRanking, reason: "Demo mode: HC improves on clean data. Skipped noise trials.", missingTeams, projectNames: projNames, noiseResults: null, hProjClean, hcProjClean, noiseTrials, aiSet };
  }

  // 4. Noise Trial Logic
  const numOpponents = teamNames.length - 1;
  const activeOpponents = Math.max(1, numOpponents - missingTeams.length);
  const cLight = Math.max(1, Math.round(activeOpponents * (1 / 14)));
  const cMed1 = Math.max(1, Math.round(activeOpponents * (2 / 14)));
  const cMed2 = Math.max(1, Math.round(activeOpponents * (3 / 14)));
  const cHeavy = Math.max(1, Math.round(activeOpponents * (5 / 14)));

  const configs = [
    [cLight, 2, `Light (${cLight} team${cLight !== 1 ? "s" : ""}, 2 swaps)`],
    [cMed1, 4, `Medium (${cMed1} team${cMed1 !== 1 ? "s" : ""}, 4 swaps)`],
    [cMed2, 4, `Medium (${cMed2} team${cMed2 !== 1 ? "s" : ""}, 4 swaps)`],
    [cHeavy, 4, `Heavy (${cHeavy} team${cHeavy !== 1 ? "s" : ""}, 4 swaps)`],
    [cHeavy, 8, `Heavy (${cHeavy} team${cHeavy !== 1 ? "s" : ""}, 8 swaps)`]
  ];

  const rng = new SeededRandom(seed ?? 42);
  const rows = [];

  for (const [numNoisy, numSwaps, label] of configs) {
    let hAi = 0, hcAi = 0, hcWins = 0, hWins = 0;
    const hPrefs = [], hcPrefs = [];
    
    for (let t = 0; t < noiseTrials; t++) {
      // We pass cleanData here so noise is injected into actual existing rankings
      const noisy = injectNoise(cleanData, numNoisy, numSwaps, rng, teamNames, myTeamName, missingTeams);
      
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
      label, 
      hAiPct: (hAi / noiseTrials) * 100, 
      hcAiPct: (hcAi / noiseTrials) * 100, 
      hMean: hPrefs.reduce((a, b) => a + b, 0) / hPrefs.length, 
      hcMean: hcPrefs.reduce((a, b) => a + b, 0) / hcPrefs.length, 
      hcWins, 
      hWins, 
      hcCatPct: ((noiseTrials - hcAi) / noiseTrials) * 100, 
      hCatPct: ((noiseTrials - hAi) / noiseTrials) * 100 
    });
  }

  // 5. Verdict Logic
  const realistic = rows[2], heavy = rows[4];
  const hcCatRealistic = realistic.hcCatPct, hCatRealistic = realistic.hCatPct, hcWinsRealistic = realistic.hcWins;
  let verdict, submit, reason;

  if (missingTeams.length) {
    if (hcCatRealistic > 10) { 
      verdict = "UNSAFE"; submit = "honest"; reason = `HC collapses under noise AND ${missingTeams.length} teams missing.`; 
    } else if (hcCatRealistic > hCatRealistic + 5) { 
      verdict = "UNSAFE"; submit = "honest"; reason = `HC catastrophe rate (${hcCatRealistic.toFixed(0)}%) exceeds honest (${hCatRealistic.toFixed(0)}%) with missing teams.`; 
    } else if (hcWinsRealistic < noiseTrials * 0.4) { 
      verdict = "RISKY"; submit = "honest"; reason = `HC wins only ${hcWinsRealistic}/${noiseTrials} under realistic noise. Conservative due to missing teams.`; 
    } else { 
      verdict = "RISKY"; submit = "honest"; reason = `HC shows improvement but ${missingTeams.length} teams missing makes optimization unreliable.`; 
    }
  } else {
    if (hcCatRealistic <= hCatRealistic + 5 && hcWinsRealistic >= noiseTrials * 0.5) { 
      verdict = "SAFE"; submit = "hc"; reason = `HC improvement survives noise. Realistic noise: HC AI=${realistic.hcAiPct.toFixed(0)}% vs Honest AI=${realistic.hAiPct.toFixed(0)}%, HC wins ${hcWinsRealistic}/${noiseTrials}.`; 
    } else if (hcCatRealistic <= hCatRealistic + 15) { 
      verdict = "RISKY"; submit = "honest"; reason = `HC improvement is fragile. Realistic noise: HC catastrophe=${hcCatRealistic.toFixed(0)}% vs Honest catastrophe=${hCatRealistic.toFixed(0)}%. Honest wins ${realistic.hWins}/${noiseTrials}.`; 
    } else { 
      verdict = "UNSAFE"; submit = "honest"; reason = `HC ranking collapses under noise. Realistic noise: HC catastrophe=${hcCatRealistic.toFixed(0)}% vs Honest catastrophe=${hCatRealistic.toFixed(0)}%. Honest wins ${realistic.hWins}/${noiseTrials}.`; 
    }
  }

  if (heavy.hWins > noiseTrials * 0.7) { 
    verdict = "UNSAFE"; submit = "honest"; reason = `HC collapses under heavy noise. Honest wins ${heavy.hWins}/${noiseTrials}. The improvement is a mirage from overfitting to perfect info.`; 
  }

  return { verdict, submit, honestRanking, hcRanking, reason, missingTeams, projectNames: projNames, noiseResults: rows, hProjClean, hcProjClean, noiseTrials, aiSet };
}
// ==================== DEFAULTS ====================
const REAL_N_PROJECTS = 32;
const REAL_N_TEAMS = 18;
const DEFAULT_TEAM_LIST = [...Array(REAL_N_TEAMS - 1)].map((_, i) => `Team_${String(i + 1).padStart(2, "0")}`).concat(["MY_TEAM"]);
const DEFAULT_PROJECT_LIST = [...Array(REAL_N_PROJECTS)].map((_, i) => `Proj_${String(i + 1).padStart(2, "0")}`);
const DEFAULT_TRUE_PREF = [...Array(REAL_N_PROJECTS)].map((_, i) => i);

// ==================== APP ====================
export default function App() {
  const myTeamName = "MY_TEAM";
  const [hydrated, setHydrated] = useState(false);
  const [syncStatus, setSyncStatus] = useState("loading");
  const [projectNames, setProjectNames] = useState([...DEFAULT_PROJECT_LIST]);
  const [teamNames, setTeamNames] = useState([...DEFAULT_TEAM_LIST]);
  const [myTruePref, setMyTruePref] = useState([...DEFAULT_TRUE_PREF]);
  const [aiProjectIndices, setAiProjectIndices] = useState(new Set());
  const [publicRankings, setPublicRankings] = useState({});
  const [jsonInput, setJsonInput] = useState("{\n  \n}");
  const [templateCount, setTemplateCount] = useState(18);
  const [noiseTrials, setNoiseTrials] = useState(100);
  const [seed, setSeed] = useState(42);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [demoMode, setDemoMode] = useState(true);
  const [dragOverIndex, setDragOverIndex] = useState(null);
  const [jsonStatus, setJsonStatus] = useState({ kind: "idle", message: "ready" });
  const [opponentTab, setOpponentTab] = useState("json");
  const [manualTeam, setManualTeam] = useState("");
  const [manualRanking, setManualRanking] = useState("");
  const [manualError, setManualError] = useState("");
  const [copyFeedback, setCopyFeedback] = useState("");
  const [importOpen, setImportOpen] = useState(false);
  const [importText, setImportText] = useState("");
  const [importError, setImportError] = useState("");

  // ── Load from Firestore on mount ──
  useEffect(() => {
    loadFromFirestore().then((data) => {
      if (data) {
        if (Array.isArray(data.projectNames) && data.projectNames.length) setProjectNames(data.projectNames);
        if (Array.isArray(data.teamNames) && data.teamNames.length) setTeamNames(data.teamNames);
        if (Array.isArray(data.myTruePref) && data.myTruePref.length) setMyTruePref(data.myTruePref);
        if (Array.isArray(data.aiProjectIndices)) setAiProjectIndices(new Set(data.aiProjectIndices));
        if (data.publicRankings && typeof data.publicRankings === "object") {
          setPublicRankings(data.publicRankings);
          const keys = Object.keys(data.publicRankings);
          if (keys.length > 0) {
            setJsonInput(JSON.stringify(data.publicRankings, null, 2));
            setJsonStatus({ kind: "ok", message: `valid · ${keys.length} team${keys.length === 1 ? "" : "s"}` });
          }
        }
      }
      setHydrated(true);
      setSyncStatus("synced");
    }).catch(() => { setHydrated(true); setSyncStatus("error"); });
  }, []);

  // ── Debounced Firestore save (merges patches) ──
  const debouncedSave = useDebouncedMerge(async (patch) => {
    setSyncStatus("saving");
    await saveToFirestore(patch);
    setSyncStatus("synced");
  }, 1200);

  // ── Wrapped setters ──
  const setAndSave = useCallback((setter, key, transform) => (updater) => {
    setter((prev) => {
      const next = typeof updater === "function" ? updater(prev) : updater;
      debouncedSave({ [key]: transform ? transform(next) : next });
      return next;
    });
  }, [debouncedSave]);

  const setAndSaveProjectNames = setAndSave(setProjectNames, "projectNames");
  const setAndSaveTeamNames = setAndSave(setTeamNames, "teamNames");
  const setAndSaveMyTruePref = setAndSave(setMyTruePref, "myTruePref");
  const setAndSaveAiProjectIndices = setAndSave(setAiProjectIndices, "aiProjectIndices", (s) => Array.from(s));

  const setAndSavePublicRankings = useCallback((newRankings) => {
    setPublicRankings(newRankings);
    debouncedSave({ publicRankings: newRankings });
  }, [debouncedSave]);

  const sortTeamNames = (arr) =>
    [...arr].sort((a, b) => {
      if (a === myTeamName) return 1;
      if (b === myTeamName) return -1;
      return a.localeCompare(b, undefined, { numeric: true });
    });

  // ── JSON editor ──
  const handleJsonInputChange = useCallback((nextJson, nextParsedOverride) => {
    setJsonInput(nextJson);
    let parsed = nextParsedOverride;
    if (parsed === undefined) {
      if (!nextJson.trim() || nextJson.trim() === "{}" || nextJson.trim() === "{\n  \n}") {
        setJsonStatus({ kind: "idle", message: "empty" });
        setAndSavePublicRankings({});
        return;
      }
      try { parsed = JSON.parse(nextJson); }
      catch (e) { setJsonStatus({ kind: "err", message: e.message }); return; }
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) { setJsonStatus({ kind: "err", message: "must be an object {team: [ranks]}" }); return; }
    }
    const detectedN = Object.values(parsed)[0]?.length || REAL_N_PROJECTS;
    setAndSaveProjectNames((prev) => {
      if (detectedN <= prev.length) return prev;
      return [...prev, ...Array.from({ length: detectedN - prev.length }, (_, i) => `Proj_${String(prev.length + i + 1).padStart(2, "0")}`)];
    });
    setAndSaveMyTruePref((prev) => {
      if (detectedN <= prev.length) return prev;
      const cur = new Set(prev), extra = [];
      for (let i = 0; i < detectedN; i++) if (!cur.has(i)) extra.push(i);
      return [...prev, ...extra];
    });
    setAndSaveTeamNames((prev) => {
      const nextSet = new Set(prev);
      Object.keys(parsed).forEach((t) => nextSet.add(t));
      nextSet.add(myTeamName);
      return sortTeamNames(Array.from(nextSet));
    });
    setAndSavePublicRankings(parsed);
    const keys = Object.keys(parsed);
    setJsonStatus({ kind: "ok", message: `valid · ${keys.length} team${keys.length === 1 ? "" : "s"}` });
  }, [setAndSaveProjectNames, setAndSaveMyTruePref, setAndSaveTeamNames, setAndSavePublicRankings]);

  const syncRankingsToJson = (newRankings) => {
    setJsonInput(JSON.stringify(newRankings, null, 2));
    const keys = Object.keys(newRankings);
    setJsonStatus({ kind: "ok", message: `valid · ${keys.length} team${keys.length === 1 ? "" : "s"}` });
  };

  // ── Manual entry ──
  const handleManualAdd = () => {
    setManualError("");
    const team = manualTeam.trim();
    if (!team) { setManualError("Team name is required."); return; }
    if (team === myTeamName) { setManualError("Cannot enter rankings for MY_TEAM."); return; }
    let rankArr;
    try { rankArr = JSON.parse(manualRanking.trim()); }
    catch { rankArr = manualRanking.trim().split(/[\s,]+/).map(Number); }
    if (!Array.isArray(rankArr) || rankArr.some(isNaN)) { setManualError("Invalid ranking — enter a JSON array or comma-separated numbers."); return; }
    const n = projectNames.length;
    if (rankArr.length !== n) { setManualError(`Expected ${n} entries, got ${rankArr.length}.`); return; }
    if (![...rankArr].sort((a, b) => a - b).every((v, i) => v === i + 1)) { setManualError(`Must be a valid 1..${n} permutation.`); return; }
    const newRankings = { ...publicRankings, [team]: rankArr };
    setAndSavePublicRankings(newRankings);
    syncRankingsToJson(newRankings);
    setAndSaveTeamNames((prev) => {
      if (prev.includes(team)) return prev;
      return sortTeamNames([...prev, team]);
    });
    setManualRanking(""); setManualTeam("");
  };

  const handleManualRemove = (team) => {
    const newRankings = { ...publicRankings };
    delete newRankings[team];
    setAndSavePublicRankings(newRankings);
    syncRankingsToJson(newRankings);
  };

  const handleManualFillRandom = () => {
    const rng = new SeededRandom(Date.now());
    setManualRanking(JSON.stringify(rng.shuffle(Array.from({ length: projectNames.length }, (_, i) => i + 1))));
  };

  const movePref = useCallback((from, to) => {
    setAndSaveMyTruePref((prev) => {
      if (from === to || to < 0 || to >= prev.length) return prev;
      const next = [...prev];
      const [item] = next.splice(from, 1);
      next.splice(to, 0, item);
      return next;
    });
  }, [setAndSaveMyTruePref]);

  const toggleAi = useCallback((projIdx) => {
    setAndSaveAiProjectIndices((prev) => {
      const next = new Set(prev);
      if (next.has(projIdx)) next.delete(projIdx); else next.add(projIdx);
      return next;
    });
  }, [setAndSaveAiProjectIndices]);

  const renameProject = useCallback((idx, newName) => {
    setAndSaveProjectNames((prev) => { const next = [...prev]; next[idx] = newName; return next; });
  }, [setAndSaveProjectNames]);

  // Parse pasted "ID: Name" lines. Numeric prefix is optional;
  // ordering of input rows determines the project index (0-based).
  const handleBulkImport = () => {
    setImportError("");
    const lines = importText.split("\n").map((l) => l.trim()).filter(Boolean);
    if (lines.length === 0) { setImportError("Paste at least one project."); return; }
    const parsed = lines.map((line) => {
      const m = line.match(/^\s*(\d+)\s*[:\-—]\s*(.+)$/);
      return m ? { id: Number(m[1]), name: m[2].trim() } : { id: null, name: line };
    });
    // Sort by external ID if every line had one; otherwise preserve paste order.
    const allHaveIds = parsed.every((p) => p.id !== null);
    const ordered = allHaveIds ? [...parsed].sort((a, b) => a.id - b.id) : parsed;
    const names = ordered.map((p) => p.name.replace(/^["']|["']$/g, ""));
    const n = names.length;
    // Reset project names, extend true pref if needed, trim if shrunk.
    setAndSaveProjectNames(() => names);
    setAndSaveMyTruePref((prev) => {
      const kept = prev.filter((idx) => idx < n);
      const present = new Set(kept);
      for (let i = 0; i < n; i++) if (!present.has(i)) kept.push(i);
      return kept;
    });
    setAndSaveAiProjectIndices((prev) => {
      const next = new Set();
      prev.forEach((idx) => { if (idx < n) next.add(idx); });
      return next;
    });
    setImportOpen(false);
    setImportText("");
  };

  const handleGenerateTemplate = () => {
    const blank = Array.from({ length: projectNames.length }, (_, i) => i + 1);
    const template = {};
    for (let i = 1; i <= templateCount; i++) template[`Team_${String(i).padStart(2, "0")}`] = [...blank];
    handleJsonInputChange(JSON.stringify(template, null, 2), template);
  };

  const runAnalysis = () => {
    const n = projectNames.length;
    for (const [team, r] of Object.entries(publicRankings)) {
      if (r.length !== n) { alert(`${team} ranking has ${r.length} entries, expected ${n}`); return; }
      if (![...r].sort((a, b) => a - b).every((v, i) => v === i + 1)) { alert(`${team} ranking is not a valid 1..${n} permutation`); return; }
    }
    setLoading(true);
    setTimeout(() => {
      const check = robustnessCheck({ myTruePreferences: myTruePref, publicRankings, aiProjectIndices: Array.from(aiProjectIndices), teamNames, projectNames, myTeamName, noiseTrials, seed, skipNoise: demoMode });
      setResult(check);
      setLoading(false);
    }, 50);
  };

  const copyRanking = (ranking) => {
    navigator.clipboard.writeText(JSON.stringify(ranking));
    setCopyFeedback("Copied!"); setTimeout(() => setCopyFeedback(""), 1800);
  };

  const handleDragStart = (e, index) => { e.dataTransfer.setData("text/plain", String(index)); e.dataTransfer.effectAllowed = "move"; };
  const handleDragOver = (e, index) => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; setDragOverIndex(index); };
  const handleDragLeave = () => setDragOverIndex(null);
  const handleDrop = (e, toIndex) => { e.preventDefault(); movePref(Number(e.dataTransfer.getData("text/plain")), toIndex); setDragOverIndex(null); };

  const otherTeams = teamNames.filter((t) => t !== myTeamName);
  const enteredCount = Object.keys(publicRankings).length;
  const totalOpp = otherTeams.length;
  const honestProj = result ? projectNames[result.hProjClean] : null;
  const hcProj = result ? projectNames[result.hcProjClean] : null;

  if (!hydrated) {
    return (
      <div className="app">
        <header className="app-header"><h1>Optimal Ranking</h1><p>Loading saved configuration…</p></header>
        <div className="card" style={{ textAlign: "center", padding: 48 }}>
          <div style={{ width: 32, height: 32, border: "3px solid var(--rule)", borderTopColor: "var(--indigo)", borderRadius: "50%", animation: "spin 0.8s linear infinite", margin: "0 auto 20px" }} />
          <p style={{ color: "var(--fog)", margin: 0 }}>Fetching your config from Firebase…</p>
        </div>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  return (
    <div className="app">
      <style>{`@keyframes spin { to { transform: rotate(360deg); } } .sync-spinner { width:10px;height:10px;border:2px solid var(--rule);border-top-color:var(--indigo);border-radius:50%;animation:spin 0.7s linear infinite;display:inline-block; }`}</style>
      <header className="app-header">
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
          <div>
            <h1>Optimal Ranking</h1>
            <p>Assignment strategy optimizer — Hungarian algorithm + hill climbing + robustness stress-testing</p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 7, padding: "6px 12px", background: "var(--bone)", border: "1px solid var(--rule)", borderRadius: 100, fontSize: "0.75rem", fontWeight: 600, color: "var(--fog)", whiteSpace: "nowrap", flexShrink: 0, marginTop: 4 }}>
            {syncStatus === "saving" ? <span className="sync-spinner" /> : <span style={{ width: 8, height: 8, borderRadius: "50%", background: syncStatus === "synced" ? "var(--olive)" : syncStatus === "error" ? "var(--ember)" : "var(--fog)", display: "inline-block" }} />}
            {syncStatus === "loading" && "Connecting…"}
            {syncStatus === "synced" && "Firebase synced"}
            {syncStatus === "saving" && "Saving…"}
            {syncStatus === "error" && "Offline"}
          </div>
        </div>
      </header>

      <section className="card">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
          <h2 style={{ margin: 0 }}>Your True Preferences</h2>
          <button className="btn btn-secondary btn-sm" onClick={() => { setImportText(""); setImportError(""); setImportOpen(true); }}>
            Import projects
          </button>
        </div>
        <p className="label" style={{ marginTop: 0 }}>Drag to reorder · double-click a name to rename · auto-saves to Firebase</p>
        <div className="pref-list">
          {myTruePref.map((projIdx, rank) => (
            <PrefRow key={projIdx} projIdx={projIdx} rank={rank} name={projectNames[projIdx]} isAi={aiProjectIndices.has(projIdx)} isDragOver={dragOverIndex === rank} onRename={(newName) => renameProject(projIdx, newName)} onDragStart={(e) => handleDragStart(e, rank)} onDragOver={(e) => handleDragOver(e, rank)} onDragLeave={handleDragLeave} onDrop={(e) => handleDrop(e, rank)} />
          ))}
        </div>
      </section>

      <section className="card">
        <h2>AI-Capable Projects</h2>
        <p className="label" style={{ marginTop: -8 }}>Toggle which projects require AI</p>
        <div className="ai-grid">
          {projectNames.map((name, idx) => (
            <button key={idx} className={`ai-chip ${aiProjectIndices.has(idx) ? "active" : ""}`} onClick={() => toggleAi(idx)}>{name}</button>
          ))}
        </div>
      </section>

      <section className="card">
        <h2>Opponent Rankings</h2>
        <div className="mb-4">
          <div className="tracker-header">
            <p className="label" style={{ margin: 0 }}>Submission Tracker</p>
            <span className="tracker-count">{enteredCount} / {totalOpp} entered</span>
          </div>
          <div className="tracker-progress">
            <div className="tracker-progress-fill" style={{ width: totalOpp > 0 ? `${(enteredCount / totalOpp) * 100}%` : "0%" }} />
          </div>
          <div className="tracker-grid">
            {otherTeams.map((team, idx) => {
              // Generate the name the watcher uses (Anonymous_1, Anonymous_2, etc.)
              const anonName = `Anonymous_${idx + 1}`;
              
              // Light up the pill if either the manual name OR the anonymous name exists
              const isSubmitted = !!(publicRankings[team] || publicRankings[anonName]);
              
              return (
                <div key={team} className={`tracker-pill ${isSubmitted ? "submitted" : "waiting"}`}>
                  <span className="tracker-dot" />
                  {isSubmitted && publicRankings[anonName] ? anonName : team}
                </div>
              );
            })}
          </div>
        </div>
        <div className="tab-bar">
          <button className={`tab-btn ${opponentTab === "json" ? "active" : ""}`} onClick={() => setOpponentTab("json")}>JSON Editor</button>
          <button className={`tab-btn ${opponentTab === "manual" ? "active" : ""}`} onClick={() => setOpponentTab("manual")}>Manual Entry</button>
        </div>
        {opponentTab === "json" && (
          <>
            <div className="flex justify-between items-end mb-2 mt-4">
              <p className="label" style={{ margin: 0 }}>JSON Input</p>
              <div className="flex gap-2 items-center text-fog" style={{ fontSize: "0.85rem" }}>
                Template for:
                <input type="number" className="template-input" value={templateCount} onChange={(e) => setTemplateCount(Math.max(1, Number(e.target.value)))} />
                teams
                <button className="btn btn-secondary btn-sm" onClick={handleGenerateTemplate}>Generate</button>
              </div>
            </div>
            <JsonEditor value={jsonInput} onChange={handleJsonInputChange} minRows={6} placeholder={`{\n  "Team_01": [3, 1, 2, 5, 4, ...],\n  "Team_02": [2, 5, 1, ...]\n}`} />
            <div className="flex justify-between items-center mt-2">
              <span className="text-fog mono" style={{ fontSize: "0.78rem" }}>{enteredCount} teams · auto-parsed on edit</span>
              <span className={`json-status-pill ${jsonStatus.kind}`}><span className="json-status-dot" />{jsonStatus.message}</span>
            </div>
          </>
        )}
        {opponentTab === "manual" && (
          <div className="manual-entry">
            <div className="manual-form">
              <div className="manual-form-row">
                <div className="field" style={{ flex: "0 0 160px" }}>
                  <p className="label">Team Name</p>
                  <input type="text" placeholder="Team_01" value={manualTeam} onChange={(e) => setManualTeam(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") handleManualAdd(); }} />
                </div>
                <div className="field" style={{ flex: 1 }}>
                  <div className="flex justify-between items-end">
                    <p className="label" style={{ margin: 0 }}>Ranking (array or comma-separated)</p>
                    <button className="btn btn-secondary btn-sm" style={{ marginBottom: 8 }} onClick={handleManualFillRandom}>Random Fill</button>
                  </div>
                  <input type="text" placeholder={`[1, 2, 3, ..., ${projectNames.length}]`} value={manualRanking} onChange={(e) => setManualRanking(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") handleManualAdd(); }} className="mono" />
                </div>
                <button className="btn btn-primary btn-sm" style={{ alignSelf: "flex-end", marginBottom: 1 }} onClick={handleManualAdd}>Add / Update</button>
              </div>
              {manualError && <p className="manual-error">{manualError}</p>}
            </div>
            {enteredCount === 0 ? (
              <p className="manual-empty">No opponent rankings entered yet.</p>
            ) : (
              <div className="manual-list">
                {Object.entries(publicRankings).map(([team, ranks]) => (
                  <div key={team} className="manual-row">
                    <span className="manual-team-name">{team}</span>
                    <span className="manual-rank-preview mono">[{ranks.slice(0, 8).join(", ")}{ranks.length > 8 ? ", …" : ""}]</span>
                    <div className="manual-actions">
                      <button className="btn btn-secondary btn-sm" onClick={() => { setManualTeam(team); setManualRanking(JSON.stringify(ranks)); setManualError(""); }}>Edit</button>
                      <button className="btn btn-sm manual-remove-btn" onClick={() => handleManualRemove(team)}>✕</button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </section>

      <section className="card">
        <h2>Settings</h2>
        <div className="result-grid">
          <div><p className="label">Noise Trials</p><input type="number" value={noiseTrials} onChange={(e) => setNoiseTrials(Number(e.target.value))} /></div>
          <div><p className="label">Random Seed</p><input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></div>
        </div>
        <div className="mode-toggle-row">
          <button className={`btn btn-sm ${demoMode ? "btn-primary" : "btn-secondary"}`} onClick={() => setDemoMode(true)}>Demo Mode</button>
          <button className={`btn btn-sm ${!demoMode ? "btn-primary" : "btn-secondary"}`} onClick={() => setDemoMode(false)}>Real Data</button>
          <span className="text-fog" style={{ fontSize: "0.85rem" }}>{demoMode ? "Skips noise trials — instant results for testing" : "Full robustness check with noise stress test"}</span>
        </div>
      </section>

      <button className="btn btn-primary w-full mt-4" onClick={runAnalysis} disabled={loading}>
        {loading ? "Running Hungarian + Hill Climb…" : demoMode ? "Run Quick Check" : "Run Robustness Check"}
      </button>

      {result && (
        <div style={{ marginTop: 32 }}>
          <div className={`verdict-banner verdict-${result.verdict.toLowerCase()}`}>
            <h2>
              {result.verdict === "SAFE" && "✓ SAFE — Submit HC Ranking"}
              {result.verdict === "RISKY" && "⚠ RISKY — Submit Honest Ranking"}
              {result.verdict === "UNSAFE" && "✗ UNSAFE — Submit Honest Ranking"}
            </h2>
            <p style={{ margin: "8px 0 0", maxWidth: 720 }}>{result.reason}</p>
            {result.missingTeams.length > 0 && <p className="text-fog" style={{ marginTop: 8, fontSize: "0.85rem" }}>Missing teams: {result.missingTeams.join(", ")}</p>}
          </div>
          <div className="result-grid">
            <div className="result-card">
              <h4>Honest Outcome</h4>
              <div className="result-project">{honestProj}</div>
              <div className="result-meta mono">{result.aiSet.has(result.hProjClean) ? "🤖 AI Project" : "Non-AI"} · True pref #{myTruePref.indexOf(result.hProjClean) + 1}</div>
            </div>
            <div className="result-card">
              <h4>HC Outcome</h4>
              <div className="result-project">{hcProj}</div>
              <div className="result-meta mono">{result.aiSet.has(result.hcProjClean) ? "🤖 AI Project" : "Non-AI"} · True pref #{myTruePref.indexOf(result.hcProjClean) + 1}</div>
            </div>
          </div>
          <div className={`submit-block ${result.submit === "honest" ? "safe" : ""}`}>
            <div className="flex justify-between items-center">
              <h3 style={{ fontFamily: "Newsreader, serif" }}>{result.submit === "hc" ? "Submit: HC Ranking" : "Submit: Honest Ranking"}</h3>
              <button className="btn btn-sm btn-primary" onClick={() => copyRanking(result.submit === "hc" ? result.hcRanking : result.honestRanking)}>{copyFeedback || "Copy"}</button>
            </div>
            <div className="code-block mono">{JSON.stringify(result.submit === "hc" ? result.hcRanking : result.honestRanking)}</div>
          </div>
          {result.noiseResults && (
            <div className="card mt-4">
              <h2>Noise Stress Test</h2>
              <div style={{ overflowX: "auto" }}>
                <table className="data-table mono">
                  <thead><tr><th>Scenario</th><th>Hon AI%</th><th>HC AI%</th><th>HC Wins</th><th>Hon Wins</th><th>HC Cat%</th><th>Hon Cat%</th></tr></thead>
                  <tbody>
                    {result.noiseResults.map((row, i) => (
                      <tr key={i}>
                        <td style={{ fontFamily: "Inter Tight, sans-serif" }}>{row.label}</td>
                        <td>{row.hAiPct.toFixed(0)}%</td><td>{row.hcAiPct.toFixed(0)}%</td>
                        <td className={row.hcWins > row.hWins ? "text-olive" : ""}>{row.hcWins}/{result.noiseTrials}</td>
                        <td className={row.hWins > row.hcWins ? "text-ember" : ""}>{row.hWins}/{result.noiseTrials}</td>
                        <td>{row.hcCatPct.toFixed(0)}%</td><td>{row.hCatPct.toFixed(0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {result.submit === "hc" && (
            <div className="card mt-4">
              <h2>What Changed vs Honest</h2>
              <div style={{ overflowX: "auto" }}>
                <table className="data-table mono">
                  <thead><tr><th style={{ fontFamily: "Inter Tight, sans-serif" }}>Project</th><th>Honest</th><th>HC</th><th>Change</th></tr></thead>
                  <tbody>
                    {projectNames.map((name, idx) => {
                      const h = result.honestRanking[idx], hc = result.hcRanking[idx];
                      if (h === hc) return null;
                      const delta = hc - h;
                      return (
                        <tr key={idx}>
                          <td style={{ fontFamily: "Inter Tight, sans-serif" }}>{name}{result.aiSet.has(idx) && <span className="ai-badge" style={{ marginLeft: 8 }}>AI</span>}</td>
                          <td>{h}</td><td>{hc}</td>
                          <td className={delta > 0 ? "text-ember" : "text-olive"}>{delta > 0 ? `↓ ${delta}` : `↑ ${Math.abs(delta)}`}</td>
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

      {importOpen && (
        <div onClick={() => setImportOpen(false)} style={{ position: "fixed", inset: 0, background: "rgba(14,17,22,0.45)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, padding: 20 }}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "var(--paper)", border: "1px solid var(--rule)", borderRadius: 12, padding: 24, maxWidth: 640, width: "100%", maxHeight: "85vh", display: "flex", flexDirection: "column", boxShadow: "0 20px 60px rgba(14,17,22,0.25)" }}>
            <h2 style={{ marginBottom: 8 }}>Import Projects</h2>
            <p className="label" style={{ marginTop: 0, marginBottom: 12 }}>
              Paste one project per line. Format: <span className="mono">ID: Name</span> or just the name. Rows will be sorted by ID if all have one.
            </p>
            <textarea
              value={importText}
              onChange={(e) => setImportText(e.target.value)}
              placeholder={`1: Type Safe LLM Library\n2: Advising Chat Bot\n3: Moodle Date Editing\n...`}
              className="mono"
              style={{ flex: 1, minHeight: 280, fontSize: "0.82rem", lineHeight: 1.5 }}
            />
            {importError && <p className="manual-error" style={{ marginTop: 8 }}>{importError}</p>}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 16, gap: 12 }}>
              <span className="text-fog" style={{ fontSize: "0.78rem" }}>
                Replaces all project names · resets AI flags · keeps your preference order where possible
              </span>
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn btn-secondary btn-sm" onClick={() => setImportOpen(false)}>Cancel</button>
                <button className="btn btn-primary btn-sm" onClick={handleBulkImport}>Import</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ==================== PREF ROW ====================
function PrefRow({ rank, name, isAi, isDragOver, onRename, onDragStart, onDragOver, onDragLeave, onDrop }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(name);
  const startEdit = () => { setDraft(name); setEditing(true); };
  const commit = () => { const t = draft.trim(); if (t && t !== name) onRename(t); setEditing(false); };
  const cancel = () => { setDraft(name); setEditing(false); };
  return (
    <div className={`pref-item ${isDragOver ? "drag-over" : ""} ${editing ? "editing" : ""}`} draggable={!editing} onDragStart={onDragStart} onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}>
      <span className="pref-rank mono">{rank + 1}</span>
      {editing ? (
        <input type="text" autoFocus className="pref-name-edit" value={draft} onChange={(e) => setDraft(e.target.value)} onBlur={commit} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); commit(); } else if (e.key === "Escape") { e.preventDefault(); cancel(); } }} onClick={(e) => e.stopPropagation()} />
      ) : (
        <span className="pref-name" onDoubleClick={startEdit} title="Double-click to rename">{name}</span>
      )}
      {isAi && <span className="ai-badge">AI</span>}
      <span className="drag-handle" title="Drag to reorder">⋮⋮</span>
    </div>
  );
}

// ==================== JSON EDITOR ====================
function escapeHtml(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function highlightJson(text) {
  const safe = escapeHtml(text); let out = "", i = 0; const n = safe.length;
  while (i < n) {
    const c = safe[i];
    if (c === '"') {
      let j = i + 1;
      while (j < n) { if (safe[j] === "\\" && j + 1 < n) { j += 2; continue; } if (safe[j] === '"') break; j++; }
      const str = safe.slice(i, j + 1); let k = j + 1;
      while (k < n && (safe[k] === " " || safe[k] === "\t")) k++;
      out += `<span class="${safe[k] === ":" ? "tok-key" : "tok-string"}">${str}</span>`; i = j + 1;
    } else if ((c === "-" || (c >= "0" && c <= "9")) && (i === 0 || /[\s,[{:]/.test(safe[i - 1]))) {
      let j = i; if (safe[j] === "-") j++;
      while (j < n && /[0-9.eE+-]/.test(safe[j])) j++;
      out += `<span class="tok-number">${safe.slice(i, j)}</span>`; i = j;
    } else if ("{}[]".includes(c)) { out += `<span class="tok-bracket">${c}</span>`; i++; }
    else if (c === "," || c === ":") { out += `<span class="tok-punct">${c}</span>`; i++; }
    else if (c === "t" && safe.slice(i, i + 4) === "true") { out += `<span class="tok-literal">true</span>`; i += 4; }
    else if (c === "f" && safe.slice(i, i + 5) === "false") { out += `<span class="tok-literal">false</span>`; i += 5; }
    else if (c === "n" && safe.slice(i, i + 4) === "null") { out += `<span class="tok-literal">null</span>`; i += 4; }
    else { out += c; i++; }
  }
  return out;
}
function JsonEditor({ value, onChange, placeholder, minRows = 6 }) {
  const lineCount = Math.max(value.split("\n").length, minRows);
  const lineNumbers = Array.from({ length: lineCount }, (_, i) => i + 1).join("\n");
  const handleScroll = (e) => {
    const overlay = e.target.parentElement.querySelector(".json-highlight");
    const gutter = e.target.parentElement.parentElement.querySelector(".json-gutter");
    if (overlay) { overlay.scrollTop = e.target.scrollTop; overlay.scrollLeft = e.target.scrollLeft; }
    if (gutter) gutter.scrollTop = e.target.scrollTop;
  };
  const handleKeyDown = (e) => {
    if (e.key === "Tab") {
      e.preventDefault();
      const ta = e.target, start = ta.selectionStart, end = ta.selectionEnd;
      const newVal = value.slice(0, start) + "  " + value.slice(end);
      onChange(newVal);
      requestAnimationFrame(() => { ta.selectionStart = ta.selectionEnd = start + 2; });
    }
  };
  return (
    <div className="json-editor">
      <div className="json-editor-body">
        <pre className="json-gutter">{lineNumbers}</pre>
        <div className="json-editor-content">
          <pre className="json-highlight" dangerouslySetInnerHTML={{ __html: highlightJson(value || "") + "\n" }} aria-hidden="true" />
          <textarea value={value} onChange={(e) => onChange(e.target.value)} onScroll={handleScroll} onKeyDown={handleKeyDown} placeholder={placeholder} spellCheck={false} autoCorrect="off" autoCapitalize="off" rows={minRows} />
        </div>
      </div>
    </div>
  );
}