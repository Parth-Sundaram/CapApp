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

// ==================== SIMULATION (identity-free) ====================
// Instead of mapping by team name, we pass a flat array of opponent rankings.
// `myRow` is the row position my team occupies in the cost matrix. This matters
// because the Hungarian algorithm breaks ties by row order — putting me at row 0
// gave me an artificial tiebreaker advantage that the real assignment process
// won't replicate. Default `myRow = "last"` is conservative (ties go against me).
//
// myRanking: length-n permutation of 1..n
// opponentRankings: array of length-n permutations
// myRow: integer 0..opponentRankings.length, or "last" (default). Determines where
//        in the cost matrix my row is inserted relative to the opponents.
function simulate(myRanking, opponentRankings, myRow = "last") {
  const oppCount = opponentRankings.length;
  const myRowIdx = myRow === "last" ? oppCount : Math.max(0, Math.min(oppCount, myRow));
  const matrix = [];
  for (let i = 0; i < oppCount; i++) {
    if (matrix.length === myRowIdx) matrix.push(myRanking.map(Number));
    matrix.push(opponentRankings[i].map(Number));
  }
  if (matrix.length === myRowIdx) matrix.push(myRanking.map(Number));
  const [rowInd, colInd] = linearSumAssignment(matrix);
  const assignment = {};
  rowInd.forEach((r, idx) => { assignment[r] = colInd[idx]; });
  return assignment[myRowIdx];
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

// Generate a uniform random permutation of 1..n
function randomPermutation(n, rng) {
  return rng.shuffle(Array.from({ length: n }, (_, i) => i + 1));
}

// Generate a "clustered" ranking by sampling each project's rank from the
// observed marginals across known rankings. Uses Plackett-Luce-style
// sequential sampling so the result is still a valid permutation.
//
// For each project, build a weight vector over ranks 1..n based on how often
// known teams placed it at each rank. Then sample rank assignments greedily,
// breaking ties by sequential constraint satisfaction.
function clusteredPermutation(n, knownRankings, rng) {
  if (knownRankings.length === 0) return randomPermutation(n, rng);
  // For each project p, count how often known teams put it at each rank.
  const marginals = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const ranking of knownRankings) {
    for (let p = 0; p < n; p++) {
      const r = ranking[p];
      if (r >= 1 && r <= n) marginals[p][r - 1] += 1;
    }
  }
  // Add Laplace smoothing so unobserved ranks still have nonzero probability
  for (let p = 0; p < n; p++) {
    for (let r = 0; r < n; r++) marginals[p][r] += 0.5;
  }
  // Greedy sampling: project order is randomized; for each project, sample a
  // rank from its (renormalized) marginal restricted to unused ranks.
  const projOrder = rng.shuffle(Array.from({ length: n }, (_, i) => i));
  const result = new Array(n).fill(0);
  const usedRanks = new Set();
  for (const p of projOrder) {
    const weights = marginals[p].map((w, r) => usedRanks.has(r + 1) ? 0 : w);
    const total = weights.reduce((a, b) => a + b, 0);
    if (total <= 0) {
      // Fallback: pick any unused rank uniformly
      const free = [];
      for (let r = 1; r <= n; r++) if (!usedRanks.has(r)) free.push(r);
      const pick = free[Math.floor(rng.random() * free.length)];
      result[p] = pick;
      usedRanks.add(pick);
    } else {
      let target = rng.random() * total;
      let pickedR = -1;
      for (let r = 0; r < n; r++) {
        target -= weights[r];
        if (target <= 0) { pickedR = r + 1; break; }
      }
      if (pickedR === -1) pickedR = n; // numerical safety
      result[p] = pickedR;
      usedRanks.add(pickedR);
    }
  }
  return result;
}

// ==================== HILL CLIMB ====================
// Now takes the array of known opponent rankings (frozen) plus an optional
// "filler" function for unknown teams. The hill climb optimizes against the
// EXPECTED outcome where unknown teams are filled with one fixed sample.
// (We use the same fixed sample for HC training; the stress test re-samples.)
function hillClimb(knownRankings, missingCount, myTruePref, aiSet, fillerFn, maxIter = 300, restarts = 5, seed = null) {
  const n = myTruePref.length;
  const rngLocal = new SeededRandom(seed ?? 42);
  const honest = makeHonest(myTruePref);

  // Single fixed fill for HC training — so the search has a stable objective.
  const fixedFill = [];
  for (let i = 0; i < missingCount; i++) fixedFill.push(fillerFn(rngLocal));
  const trainingOpponents = [...knownRankings, ...fixedFill];

  let bestScore = utility(simulate(honest, trainingOpponents), myTruePref, aiSet);
  let bestRanking = [...honest];

  for (let restart = 0; restart < restarts; restart++) {
    let current = restart === 0 ? [...honest] : randomPermutation(n, rngLocal);
    let currentScore = utility(simulate(current, trainingOpponents), myTruePref, aiSet);
    for (let it = 0; it < maxIter; it++) {
      let bestSwap = null, bestSwapScore = currentScore;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          [current[i], current[j]] = [current[j], current[i]];
          const s = utility(simulate(current, trainingOpponents), myTruePref, aiSet);
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

// ==================== ROBUSTNESS CHECK ====================
// New model: known opponent rankings are FROZEN (we have them from the
// watcher). Only unknown teams are sampled per trial. Three scenarios:
//   - Neutral: unknown teams = uniform random permutations
//   - Clustered: unknown teams sampled from observed marginals (assumes
//                preferences cluster around what we've seen)
//   - Adversarial: per trial, sample K unknown configurations and keep the
//                  worst outcome for me (models "opponents conspire against me")
function robustnessCheck({
  myTruePreferences,
  knownRankings, // array of length-n permutations, identity-free
  totalOpponents, // total number of opponent teams (excluding MY_TEAM)
  aiProjectIndices,
  projectNames,
  noiseTrials = 100,
  seed = null,
  skipNoise = false,
}) {
  const n = myTruePreferences.length;
  const aiSet = new Set(aiProjectIndices);
  const missingCount = Math.max(0, totalOpponents - knownRankings.length);

  const projNames = projectNames || Array.from({ length: n }, (_, i) => `Proj_${String(i).padStart(2, "0")}`);
  const honestRanking = makeHonest(myTruePreferences);

  // For "clean" comparison we need a representative scenario. Use neutral
  // uniform fill. Both honest and HC are evaluated against this same fill.
  const rngClean = new SeededRandom(seed ?? 42);
  const cleanFill = [];
  for (let i = 0; i < missingCount; i++) cleanFill.push(randomPermutation(n, rngClean));
  const cleanOpponents = [...knownRankings, ...cleanFill];

  const hProjClean = simulate(honestRanking, cleanOpponents);
  const hScoreClean = utility(hProjClean, myTruePreferences, aiSet);
  const [hcRanking, hcScoreClean] = hillClimb(
    knownRankings, missingCount, myTruePreferences, aiSet,
    (rng) => randomPermutation(n, rng),
    300, 5, seed
  );
  const hcProjClean = simulate(hcRanking, cleanOpponents);

  if (hcScoreClean >= hScoreClean) {
    return {
      verdict: "SAFE", submit: "honest", honestRanking, hcRanking,
      reason: "HC found no improvement over honest. Nothing to risk.",
      knownCount: knownRankings.length, missingCount, projectNames: projNames,
      noiseResults: null, hProjClean, hcProjClean, noiseTrials, aiSet,
    };
  }
  if (skipNoise) {
    return {
      verdict: "SAFE", submit: "hc", honestRanking, hcRanking,
      reason: "Demo mode: HC improves on clean data. Skipped stress trials.",
      knownCount: knownRankings.length, missingCount, projectNames: projNames,
      noiseResults: null, hProjClean, hcProjClean, noiseTrials, aiSet,
    };
  }

  // Three scenarios for the unknown-teams fill. Each ships a `description`
  // that's surfaced in the UI under the scenario label so you can see what's
  // actually being tested.
  const scenarios = [
    {
      label: "Uniform random",
      key: "neutral",
      description: "Each unknown team's ranking is a uniform-random permutation. No assumption about correlated tastes. The most pessimistic baseline that still treats unknowns as independent.",
      sampleTrial: (rng) => {
        const fill = [];
        for (let i = 0; i < missingCount; i++) fill.push(randomPermutation(n, rng));
        return [...knownRankings, ...fill];
      },
    },
    {
      label: "Following known patterns",
      key: "clustered",
      description: `Each unknown team samples a ranking from the empirical rank distribution of the ${knownRankings.length} known teams. If a project sat at rank 1 for half the known teams, unknowns rank it #1 about half the time. Models "students at the same school have correlated tastes."`,
      sampleTrial: (rng) => {
        const fill = [];
        for (let i = 0; i < missingCount; i++) fill.push(clusteredPermutation(n, knownRankings, rng));
        return [...knownRankings, ...fill];
      },
    },
    {
      label: "Targeted adversarial",
      key: "adversarial",
      description: `Every unknown team submits the SAME ranking you do, creating maximum contention for whatever you put at rank #1. Tests whether your ranking survives a worst-case world where unknowns specifically target your top picks.`,
      // sampleTrial is unused for this scenario — handled specially below.
      sampleTrial: null,
    },
  ];

  const rng = new SeededRandom(seed ?? 42);
  const rows = [];

  for (const scenario of scenarios) {
    let hAi = 0, hcAi = 0, hcWins = 0, hWins = 0;
    const hPrefs = [], hcPrefs = [];
    for (let t = 0; t < noiseTrials; t++) {
      // Pick a row position for MY_TEAM in this trial's cost matrix. Varying
      // this across trials averages out the Hungarian algorithm's row-order
      // tiebreaker bias — both honest and HC are evaluated at the SAME row
      // for fair comparison within each trial, but row position itself shifts
      // across trials.
      let hu, hcu, hp, hcp;
      const totalOppsThisTrial = knownRankings.length + missingCount;
      const myRowForTrial = Math.floor(rng.random() * (totalOppsThisTrial + 1));

      if (scenario.key === "adversarial") {
        // Each unknown team submits the SAME ranking as the candidate being
        // evaluated. For honest: unknowns mirror honest. For HC: unknowns mirror HC.
        // This is the targeted worst case — maximum contention on whatever
        // project the candidate puts at #1.
        const hOpps = [...knownRankings];
        const hcOpps = [...knownRankings];
        for (let i = 0; i < missingCount; i++) {
          hOpps.push([...honestRanking]);
          hcOpps.push([...hcRanking]);
        }
        hp = simulate(honestRanking, hOpps, myRowForTrial);
        hu = utility(hp, myTruePreferences, aiSet);
        hcp = simulate(hcRanking, hcOpps, myRowForTrial);
        hcu = utility(hcp, myTruePreferences, aiSet);
      } else {
        const opps = scenario.sampleTrial(rng);
        hp = simulate(honestRanking, opps, myRowForTrial);
        hu = utility(hp, myTruePreferences, aiSet);
        hcp = simulate(hcRanking, opps, myRowForTrial);
        hcu = utility(hcp, myTruePreferences, aiSet);
      }
      hPrefs.push(hu < 1000 ? hu : 99); if (aiSet.has(hp)) hAi++;
      hcPrefs.push(hcu < 1000 ? hcu : 99); if (aiSet.has(hcp)) hcAi++;
      if (hu < hcu) hWins++; else if (hcu < hu) hcWins++;
    }
    rows.push({
      label: scenario.label,
      key: scenario.key,
      description: scenario.description,
      hAiPct: (hAi / noiseTrials) * 100, hcAiPct: (hcAi / noiseTrials) * 100,
      hMean: hPrefs.reduce((a, b) => a + b, 0) / hPrefs.length,
      hcMean: hcPrefs.reduce((a, b) => a + b, 0) / hcPrefs.length,
      hcWins, hWins,
      hcCatPct: ((noiseTrials - hcAi) / noiseTrials) * 100,
      hCatPct: ((noiseTrials - hAi) / noiseTrials) * 100,
    });
  }

  // Verdict logic — focus on neutral + clustered + adversarial.
  // SAFE: HC wins or ties on all three scenarios (neutral, clustered) AND
  //       doesn't catastrophically lose adversarial.
  // RISKY: HC wins neutral but loses one of clustered/adversarial badly.
  // UNSAFE: HC loses adversarial badly (>70% honest wins) OR loses neutral.
  const neutral = rows.find((r) => r.key === "neutral");
  const clustered = rows.find((r) => r.key === "clustered");
  const adversarial = rows.find((r) => r.key === "adversarial");
  let verdict, submit, reason;

  const advHonestWins = adversarial.hWins;
  const advThreshold = noiseTrials * 0.6;
  const neutralOK = neutral.hcWins >= noiseTrials * 0.45 || neutral.hcMean <= neutral.hMean;
  const clusteredOK = clustered.hcWins >= noiseTrials * 0.45 || clustered.hcMean <= clustered.hMean;
  const adversarialOK = advHonestWins < advThreshold;

  if (missingCount === 0) {
    verdict = "SAFE"; submit = "hc";
    reason = `All ${knownRankings.length} opponent rankings are known. HC outcome is deterministic.`;
  } else if (neutralOK && clusteredOK && adversarialOK) {
    verdict = "SAFE"; submit = "hc";
    reason = `HC survives all three scenarios. Neutral: ${neutral.hcWins}/${noiseTrials} wins. Clustered: ${clustered.hcWins}/${noiseTrials}. Adversarial: honest wins ${advHonestWins}/${noiseTrials}.`;
  } else if (!adversarialOK) {
    verdict = "UNSAFE"; submit = "honest";
    reason = `HC collapses under adversarial scenarios. Honest wins ${advHonestWins}/${noiseTrials} when ${missingCount} unknown teams converge against you.`;
  } else if (!neutralOK) {
    verdict = "UNSAFE"; submit = "honest";
    reason = `HC loses even under neutral random unknowns: honest wins ${neutral.hWins}/${noiseTrials}. The exploit doesn't survive ${missingCount} unknown teams.`;
  } else {
    verdict = "RISKY"; submit = "honest";
    reason = `HC wins neutral (${neutral.hcWins}/${noiseTrials}) but is fragile in clustered/adversarial cases. With ${missingCount} unknown teams the edge is thin.`;
  }

  return {
    verdict, submit, honestRanking, hcRanking, reason,
    knownCount: knownRankings.length, missingCount,
    projectNames: projNames, noiseResults: rows,
    hProjClean, hcProjClean, noiseTrials, aiSet,
  };
}

// ==================== DEFAULTS ====================
const REAL_N_PROJECTS = 32;
const DEFAULT_TOTAL_TEAMS = 18;
const DEFAULT_PROJECT_LIST = [...Array(REAL_N_PROJECTS)].map((_, i) => `Proj_${String(i + 1).padStart(2, "0")}`);
const DEFAULT_TRUE_PREF = [...Array(REAL_N_PROJECTS)].map((_, i) => i);

// ==================== APP ====================
export default function App() {
  const [hydrated, setHydrated] = useState(false);
  const [syncStatus, setSyncStatus] = useState("loading");
  const [projectNames, setProjectNames] = useState([...DEFAULT_PROJECT_LIST]);
  const [myTruePref, setMyTruePref] = useState([...DEFAULT_TRUE_PREF]);
  const [aiProjectIndices, setAiProjectIndices] = useState(new Set());
  const [publicRankings, setPublicRankings] = useState({});
  const [totalTeams, setTotalTeams] = useState(DEFAULT_TOTAL_TEAMS); // includes MY_TEAM
  const [jsonInput, setJsonInput] = useState("{\n  \n}");
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
  const [prefOpen, setPrefOpen] = useState(false);
  const [prefText, setPrefText] = useState("");
  const [prefError, setPrefError] = useState("");

  // ── Load from Firestore on mount ──
  useEffect(() => {
    loadFromFirestore().then((data) => {
      if (data) {
        if (Array.isArray(data.projectNames) && data.projectNames.length) setProjectNames(data.projectNames);
        if (Array.isArray(data.myTruePref) && data.myTruePref.length) setMyTruePref(data.myTruePref);
        if (Array.isArray(data.aiProjectIndices)) setAiProjectIndices(new Set(data.aiProjectIndices));
        if (typeof data.totalTeams === "number" && data.totalTeams >= 2) setTotalTeams(data.totalTeams);
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

  // ── Debounced Firestore save ──
  const debouncedSave = useDebouncedMerge(async (patch) => {
    setSyncStatus("saving");
    await saveToFirestore(patch);
    setSyncStatus("synced");
  }, 1200);

  const setAndSave = useCallback((setter, key, transform) => (updater) => {
    setter((prev) => {
      const next = typeof updater === "function" ? updater(prev) : updater;
      debouncedSave({ [key]: transform ? transform(next) : next });
      return next;
    });
  }, [debouncedSave]);

  const setAndSaveProjectNames = setAndSave(setProjectNames, "projectNames");
  const setAndSaveMyTruePref = setAndSave(setMyTruePref, "myTruePref");
  const setAndSaveAiProjectIndices = setAndSave(setAiProjectIndices, "aiProjectIndices", (s) => Array.from(s));
  const setAndSaveTotalTeams = setAndSave(setTotalTeams, "totalTeams");

  const setAndSavePublicRankings = useCallback((newRankings) => {
    setPublicRankings(newRankings);
    debouncedSave({ publicRankings: newRankings });
  }, [debouncedSave]);

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
    setAndSavePublicRankings(parsed);
    const keys = Object.keys(parsed);
    setJsonStatus({ kind: "ok", message: `valid · ${keys.length} team${keys.length === 1 ? "" : "s"}` });
  }, [setAndSaveProjectNames, setAndSaveMyTruePref, setAndSavePublicRankings]);

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
    if (team === "MY_TEAM") { setManualError("Cannot use 'MY_TEAM' as a label."); return; }
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

  const handleBulkImport = () => {
    setImportError("");
    const lines = importText.split("\n").map((l) => l.trim()).filter(Boolean);
    if (lines.length === 0) { setImportError("Paste at least one project."); return; }
    const parsed = lines.map((line) => {
      const m = line.match(/^\s*(\d+)\s*[:\-—]\s*(.+)$/);
      return m ? { id: Number(m[1]), name: m[2].trim() } : { id: null, name: line };
    });
    const allHaveIds = parsed.every((p) => p.id !== null);
    const ordered = allHaveIds ? [...parsed].sort((a, b) => a.id - b.id) : parsed;
    const names = ordered.map((p) => p.name.replace(/^["']|["']$/g, ""));
    const n = names.length;
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

  // ── True Preferences import ──
  // Accepts the SAME semantic as opponent rankings: arr[projectIndex] = rank.
  // Two input formats auto-detected:
  //   - JSON array: [5, 1, 3, 2, ...] — value at index i is the rank for project i.
  //                 0/null/missing entries mean "unranked, append to end."
  //   - Line-based: "1: 5" or "1, 5" or "1 5" — meaning project ID 1 → rank 5.
  //                 Multiple lines, any not mentioned are unranked.
  // After parsing, we invert: build a myTruePref (preference-order array of
  // project indices), using the ranks the user gave, then append unranked
  // projects in their current order to fill remaining slots.
  const parsePreferenceInput = (text) => {
    const trimmed = text.trim();
    if (!trimmed) return { error: "Paste at least one entry." };
    const n = projectNames.length;
    // ProjectIndex → rank map (1..n). Missing or 0 entries = unranked.
    const projToRank = {};

    // Try JSON array first
    if (trimmed.startsWith("[")) {
      let arr;
      try { arr = JSON.parse(trimmed); }
      catch (e) { return { error: `JSON parse error: ${e.message}` }; }
      if (!Array.isArray(arr)) return { error: "Expected a JSON array." };
      if (arr.length > n) return { error: `Array has ${arr.length} entries; expected at most ${n}.` };
      for (let i = 0; i < arr.length; i++) {
        const r = arr[i];
        if (r === null || r === 0 || r === undefined || r === "") continue;
        const rNum = Number(r);
        if (!Number.isInteger(rNum) || rNum < 1 || rNum > n) {
          return { error: `Entry at index ${i} = ${r}; must be a positive integer ≤ ${n} (or 0/null to leave unranked).` };
        }
        projToRank[i] = rNum;
      }
    } else {
      // Line-based: each line is "projectID separator rank" or just "rank"
      // (when one-per-line and we infer position from line order).
      const lines = trimmed.split("\n").map((l) => l.trim()).filter(Boolean);
      // Detect format: does every line have a separator?
      const sep = /[:\-—,]\s*|\s+/;
      const looksPaired = lines.every((l) => /^\s*\d+\s*[:\-—,]\s*\d+\s*$/.test(l)) ||
                           lines.every((l) => /^\s*\d+\s+\d+\s*$/.test(l));
      if (looksPaired) {
        for (const l of lines) {
          const m = l.match(/^\s*(\d+)\s*[:\-—,\s]+\s*(\d+)\s*$/);
          if (!m) return { error: `Could not parse "${l}" as "ID: rank".` };
          const externalId = Number(m[1]);
          const rank = Number(m[2]);
          // Find project by external ID — we don't store this directly, so we
          // assume the user imported projects via "Import projects" which sorted
          // them by external ID into indices 0..n-1. We let the user reference
          // by 1..n (internal position) OR by external ID — try both.
          // Since project names like "1: Type Safe LLM Library" embed the ID,
          // we can sniff it from the name. Otherwise fall back to 1-based index.
          let projIdx = -1;
          for (let i = 0; i < projectNames.length; i++) {
            const m2 = projectNames[i].match(/^(\d+)\s*:/);
            if (m2 && Number(m2[1]) === externalId) { projIdx = i; break; }
          }
          if (projIdx === -1) {
            // Maybe user meant 1-based internal position
            if (externalId >= 1 && externalId <= n) projIdx = externalId - 1;
          }
          if (projIdx === -1) return { error: `No project found for ID ${externalId}.` };
          if (rank < 1 || rank > n) return { error: `Rank ${rank} out of range 1..${n}.` };
          projToRank[projIdx] = rank;
        }
      } else {
        // One value per line, taken in order: line k = rank for project index k
        if (lines.length > n) return { error: `${lines.length} entries; expected at most ${n}.` };
        for (let i = 0; i < lines.length; i++) {
          const r = Number(lines[i]);
          if (lines[i] === "" || lines[i] === "0" || r === 0) continue;
          if (!Number.isInteger(r) || r < 1 || r > n) {
            return { error: `Line ${i + 1} = "${lines[i]}"; expected a positive integer ≤ ${n}.` };
          }
          projToRank[i] = r;
        }
      }
    }

    // Validate: no duplicate ranks
    const seenRanks = new Map();
    for (const [idx, r] of Object.entries(projToRank)) {
      if (seenRanks.has(r)) {
        return { error: `Rank ${r} assigned to multiple projects (indices ${seenRanks.get(r)} and ${idx}).` };
      }
      seenRanks.set(r, idx);
    }

    return { projToRank };
  };

  const handlePrefImport = () => {
    setPrefError("");
    const parsed = parsePreferenceInput(prefText);
    if (parsed.error) { setPrefError(parsed.error); return; }
    const { projToRank } = parsed;
    const n = projectNames.length;

    // Build new myTruePref: pref-order array of project indices.
    // Step 1: place ranked projects at their assigned positions.
    // Step 2: append remaining projects (in current myTruePref order) to fill the unused ranks.
    const result = new Array(n).fill(-1);
    for (const [idxStr, r] of Object.entries(projToRank)) {
      result[r - 1] = Number(idxStr);
    }
    const usedProjects = new Set(Object.keys(projToRank).map(Number));
    // Iterate the current preference order; for any project not used, drop it into the next empty slot.
    let cursor = 0;
    for (const projIdx of myTruePref) {
      if (usedProjects.has(projIdx)) continue;
      while (cursor < n && result[cursor] !== -1) cursor++;
      if (cursor >= n) break;
      result[cursor] = projIdx;
      cursor++;
    }
    if (result.some((v) => v === -1)) {
      setPrefError("Internal error: not all slots filled. Please file a bug.");
      return;
    }
    setAndSaveMyTruePref(() => result);
    setPrefOpen(false);
    setPrefText("");
  };

  const runAnalysis = () => {
    const n = projectNames.length;
    for (const [team, r] of Object.entries(publicRankings)) {
      if (r.length !== n) { alert(`${team} ranking has ${r.length} entries, expected ${n}`); return; }
      if (![...r].sort((a, b) => a - b).every((v, i) => v === i + 1)) { alert(`${team} ranking is not a valid 1..${n} permutation`); return; }
    }
    // Identity-free: rankings are just an array of length-n permutations.
    const knownRankings = Object.values(publicRankings);
    const totalOpponents = Math.max(0, totalTeams - 1);
    if (knownRankings.length > totalOpponents) {
      alert(`You have ${knownRankings.length} opponent rankings but Total Teams is set to ${totalTeams} (only ${totalOpponents} opponents). Increase Total Teams in Settings.`);
      return;
    }
    setLoading(true);
    setTimeout(() => {
      const check = robustnessCheck({
        myTruePreferences: myTruePref,
        knownRankings,
        totalOpponents,
        aiProjectIndices: Array.from(aiProjectIndices),
        projectNames,
        noiseTrials,
        seed,
        skipNoise: demoMode,
      });
      setResult(check);
      setLoading(false);
    }, 50);
  };

  const copyOrderedList = async () => {
    if (!result) return;
    const ranking = result.submit === "hc" ? result.hcRanking : result.honestRanking;
    const indices = Array.from({ length: projectNames.length }, (_, i) => i)
      .sort((a, b) => ranking[a] - ranking[b]);
    const text = indices.map((idx, i) => `${i + 1}. ${projectNames[idx]}`).join("\n");
    try {
      // Try modern clipboard API
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        // Fallback: hidden textarea + execCommand
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed"; ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }
      setCopyFeedback("Copied!");
    } catch (e) {
      console.error("Clipboard copy failed:", e);
      setCopyFeedback("Copy failed");
    }
    setTimeout(() => setCopyFeedback(""), 1800);
  };

  const handleDragStart = (e, index) => { e.dataTransfer.setData("text/plain", String(index)); e.dataTransfer.effectAllowed = "move"; };
  const handleDragOver = (e, index) => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; setDragOverIndex(index); };
  const handleDragLeave = () => setDragOverIndex(null);
  const handleDrop = (e, toIndex) => { e.preventDefault(); movePref(Number(e.dataTransfer.getData("text/plain")), toIndex); setDragOverIndex(null); };

  const enteredCount = Object.keys(publicRankings).length;
  const totalOpp = Math.max(0, totalTeams - 1);
  const missingCount = Math.max(0, totalOpp - enteredCount);

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
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8, gap: 8 }}>
          <h2 style={{ margin: 0 }}>Your True Preferences</h2>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-secondary btn-sm" onClick={() => { setPrefText(""); setPrefError(""); setPrefOpen(true); }}>
              Paste preferences
            </button>
            <button className="btn btn-secondary btn-sm" onClick={() => { setImportText(""); setImportError(""); setImportOpen(true); }}>
              Import projects
            </button>
          </div>
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
            <span className="tracker-count">{enteredCount} known · {missingCount} unknown · {totalOpp} total opponents</span>
          </div>
          <div className="tracker-progress">
            <div className="tracker-progress-fill" style={{ width: totalOpp > 0 ? `${(enteredCount / totalOpp) * 100}%` : "0%" }} />
          </div>
          <div className="tracker-grid">
            {Object.keys(publicRankings).map((team) => (
              <div key={team} className="tracker-pill submitted">
                <span className="tracker-dot" />{team}
              </div>
            ))}
            {Array.from({ length: missingCount }, (_, i) => (
              <div key={`unknown-${i}`} className="tracker-pill waiting">
                <span className="tracker-dot" />unknown
              </div>
            ))}
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
            </div>
            <JsonEditor value={jsonInput} onChange={handleJsonInputChange} minRows={6} placeholder={`{\n  "Anonymous_1": [3, 1, 2, 5, 4, ...],\n  "Anonymous_2": [2, 5, 1, ...]\n}`} />
            <div className="flex justify-between items-center mt-2">
              <span className="text-fog mono" style={{ fontSize: "0.78rem" }}>{enteredCount} known rankings · auto-parsed on edit</span>
              <span className={`json-status-pill ${jsonStatus.kind}`}><span className="json-status-dot" />{jsonStatus.message}</span>
            </div>
          </>
        )}
        {opponentTab === "manual" && (
          <div className="manual-entry">
            <div className="manual-form">
              <div className="manual-form-row">
                <div className="field" style={{ flex: "0 0 160px" }}>
                  <p className="label">Team Label</p>
                  <input type="text" placeholder="Anonymous_1" value={manualTeam} onChange={(e) => setManualTeam(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") handleManualAdd(); }} />
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
          <div>
            <p className="label">Total Teams (including yours)</p>
            <input type="number" min={2} value={totalTeams} onChange={(e) => setAndSaveTotalTeams(Math.max(2, Number(e.target.value) || 2))} />
          </div>
          <div><p className="label">Noise Trials</p><input type="number" value={noiseTrials} onChange={(e) => setNoiseTrials(Number(e.target.value))} /></div>
          <div><p className="label">Random Seed</p><input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></div>
        </div>
        <div className="mode-toggle-row">
          <button className={`btn btn-sm ${demoMode ? "btn-primary" : "btn-secondary"}`} onClick={() => setDemoMode(true)}>Demo Mode</button>
          <button className={`btn btn-sm ${!demoMode ? "btn-primary" : "btn-secondary"}`} onClick={() => setDemoMode(false)}>Real Data</button>
          <span className="text-fog" style={{ fontSize: "0.85rem" }}>{demoMode ? "Skips stress trials — instant results for testing" : "Full robustness check with 3-scenario stress test"}</span>
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
            <p className="text-fog" style={{ marginTop: 8, fontSize: "0.85rem" }}>
              {result.knownCount} known opponent ranking{result.knownCount === 1 ? "" : "s"} · {result.missingCount} unknown
            </p>
          </div>

          <div className="result-grid">
            <div className="result-card">
              <h4>Honest Outcome</h4>
              <div className="result-project">{projectNames[result.hProjClean]}</div>
              <div className="result-meta mono">{result.aiSet.has(result.hProjClean) ? "🤖 AI Project" : "Non-AI"} · True pref #{myTruePref.indexOf(result.hProjClean) + 1}</div>
            </div>
            <div className="result-card">
              <h4>HC Outcome</h4>
              <div className="result-project">{projectNames[result.hcProjClean]}</div>
              <div className="result-meta mono">{result.aiSet.has(result.hcProjClean) ? "🤖 AI Project" : "Non-AI"} · True pref #{myTruePref.indexOf(result.hcProjClean) + 1}</div>
            </div>
          </div>

          <div className={`submit-block ${result.submit === "honest" ? "safe" : ""}`}>
            <div className="flex justify-between items-center" style={{ marginBottom: 4 }}>
              <h3 style={{ fontFamily: "Newsreader, serif", margin: 0 }}>
                Submit on Moodle: {result.submit === "hc" ? "Hill Climb Ranking" : "Honest Ranking"}
              </h3>
              <button className="btn btn-sm btn-primary" onClick={copyOrderedList}>{copyFeedback || "Copy List"}</button>
            </div>
            <p className="label" style={{ marginTop: 0, marginBottom: 12 }}>Rank projects in this exact order, top = #1</p>
            <ol style={{ paddingLeft: 24, margin: 0, columnCount: 2, columnGap: 24, fontFamily: "Inter Tight, sans-serif", fontSize: "0.9rem" }}>
              {(() => {
                const ranking = result.submit === "hc" ? result.hcRanking : result.honestRanking;
                const indices = Array.from({ length: projectNames.length }, (_, i) => i)
                  .sort((a, b) => ranking[a] - ranking[b]);
                return indices.map((projIdx) => (
                  <li key={projIdx} style={{ marginBottom: 4, breakInside: "avoid" }}>
                    {projectNames[projIdx]}
                    {result.aiSet.has(projIdx) && <span className="ai-badge" style={{ marginLeft: 8 }}>AI</span>}
                  </li>
                ));
              })()}
            </ol>
          </div>

          {/* If we're NOT submitting the HC ranking but HC differs, show what HC suggested
              so the user can see what they're "missing" and judge whether to override. */}
          {result.submit === "honest" &&
            JSON.stringify(result.honestRanking) !== JSON.stringify(result.hcRanking) && (
            <div className="card mt-4" style={{ borderLeft: "3px solid var(--ember)" }}>
              <div className="flex justify-between items-center" style={{ marginBottom: 4 }}>
                <h3 style={{ fontFamily: "Newsreader, serif", margin: 0 }}>
                  Hill Climb's alternative ranking
                </h3>
                <span className="text-fog" style={{ fontSize: "0.8rem" }}>(not recommended)</span>
              </div>
              <p className="label" style={{ marginTop: 0, marginBottom: 12 }}>
                What HC would have submitted instead — shown so you can compare.
              </p>
              <ol style={{ paddingLeft: 24, margin: 0, columnCount: 2, columnGap: 24, fontFamily: "Inter Tight, sans-serif", fontSize: "0.9rem" }}>
                {(() => {
                  const ranking = result.hcRanking;
                  const indices = Array.from({ length: projectNames.length }, (_, i) => i)
                    .sort((a, b) => ranking[a] - ranking[b]);
                  return indices.map((projIdx) => (
                    <li key={projIdx} style={{ marginBottom: 4, breakInside: "avoid" }}>
                      {projectNames[projIdx]}
                      {result.aiSet.has(projIdx) && <span className="ai-badge" style={{ marginLeft: 8 }}>AI</span>}
                    </li>
                  ));
                })()}
              </ol>
            </div>
          )}

          {result.noiseResults ? (
            <div className="card mt-4">
              <h2>Stress Test Across Unknown-Team Scenarios</h2>
              <p className="label" style={{ marginTop: -8 }}>
                {result.knownCount} known ranking{result.knownCount === 1 ? "" : "s"} frozen. Only the {result.missingCount} unknown team{result.missingCount === 1 ? "" : "s"} {result.missingCount === 1 ? "is" : "are"} sampled per trial.
              </p>
              <div style={{ overflowX: "auto" }}>
                <table className="data-table mono">
                  <thead>
                    <tr>
                      <th>Scenario</th>
                      <th>Hon AI%</th>
                      <th>HC AI%</th>
                      <th>HC Wins</th>
                      <th>Hon Wins</th>
                      <th>HC Mean Pref</th>
                      <th>Hon Mean Pref</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.noiseResults.map((row, i) => (
                      <tr key={i}>
                        <td style={{ fontFamily: "Inter Tight, sans-serif", verticalAlign: "top", maxWidth: 280 }}>
                          <div style={{ fontWeight: 600 }}>{row.label}</div>
                          {row.description && (
                            <div style={{ fontSize: "0.72rem", color: "var(--fog)", marginTop: 4, lineHeight: 1.4, fontFamily: "Inter Tight, sans-serif", whiteSpace: "normal" }}>
                              {row.description}
                            </div>
                          )}
                        </td>
                        <td style={{ verticalAlign: "top" }}>{row.hAiPct.toFixed(0)}%</td>
                        <td style={{ verticalAlign: "top" }}>{row.hcAiPct.toFixed(0)}%</td>
                        <td style={{ verticalAlign: "top" }} className={row.hcWins > row.hWins ? "text-olive" : ""}>{row.hcWins}/{result.noiseTrials}</td>
                        <td style={{ verticalAlign: "top" }} className={row.hWins > row.hcWins ? "text-ember" : ""}>{row.hWins}/{result.noiseTrials}</td>
                        <td style={{ verticalAlign: "top" }}>{row.hcMean.toFixed(2)}</td>
                        <td style={{ verticalAlign: "top" }}>{row.hMean.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-fog" style={{ fontSize: "0.78rem", marginTop: 12, marginBottom: 0, lineHeight: 1.5 }}>
                <b>Mean Pref</b> = average rank in your true preference list across all trials (lower is better — Mean Pref of 1.0 means you'd always get your #1 choice).<br />
                <b>Wins</b> = trials where that ranking gave a strictly better outcome than the other. Trials where both rankings produced the same outcome don't count for either side.<br />
                <b>AI %</b> = fraction of trials where the assigned project was AI-capable.
              </p>
            </div>
          ) : (
            <div className="card mt-4">
              <h2>Stress Test</h2>
              <p style={{ margin: 0, color: "var(--fog)" }}>
                {demoMode
                  ? "Stress test skipped — Demo Mode is on. Switch to 'Real Data' in Settings to run all three scenarios."
                  : result.missingCount === 0
                  ? "No unknown teams — all opponent rankings are known. The outcome is deterministic, no stress test needed."
                  : "Stress test not run because HC found no improvement over the honest ranking. There's nothing to compare against."}
              </p>
            </div>
          )}

          {JSON.stringify(result.honestRanking) !== JSON.stringify(result.hcRanking) && (
            <div className="card mt-4">
              <h2>{result.submit === "hc" ? "What Changed vs Honest" : "Where Honest and HC Differ"}</h2>
              <p className="label" style={{ marginTop: -8 }}>
                {result.submit === "hc"
                  ? "Positions HC moved relative to your honest ranking."
                  : "The two rankings differ at these positions. Shown so you can see HC's specific moves."}
              </p>
              <div style={{ overflowX: "auto" }}>
                <table className="data-table mono">
                  <thead><tr><th style={{ fontFamily: "Inter Tight, sans-serif" }}>Project</th><th>Honest rank</th><th>HC rank</th><th>HC's move</th></tr></thead>
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
            <textarea value={importText} onChange={(e) => setImportText(e.target.value)} placeholder={`1: Type Safe LLM Library\n2: Advising Chat Bot\n3: Moodle Date Editing\n...`} className="mono" style={{ flex: 1, minHeight: 280, fontSize: "0.82rem", lineHeight: 1.5 }} />
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

      {prefOpen && (
        <div onClick={() => setPrefOpen(false)} style={{ position: "fixed", inset: 0, background: "rgba(14,17,22,0.45)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, padding: 20 }}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "var(--paper)", border: "1px solid var(--rule)", borderRadius: 12, padding: 24, maxWidth: 680, width: "100%", maxHeight: "85vh", display: "flex", flexDirection: "column", boxShadow: "0 20px 60px rgba(14,17,22,0.25)" }}>
            <h2 style={{ marginBottom: 8 }}>Paste True Preferences</h2>
            <p className="label" style={{ marginTop: 0, marginBottom: 12 }}>
              Same semantic as opponent rankings: <span className="mono">arr[i] = rank you assign to project i</span>. Partial input is OK — unranked projects keep their current order at the end.
            </p>
            <div style={{ fontSize: "0.78rem", color: "var(--fog)", marginBottom: 12, lineHeight: 1.6 }}>
              Three accepted formats:
              <ul style={{ margin: "4px 0 0 18px", padding: 0 }}>
                <li><b>JSON array</b>: <span className="mono">[6, 12, 32, 31, ...]</span> — same as opponent ranking format</li>
                <li><b>ID: rank pairs</b>: <span className="mono">14: 1{"\n"}12: 2{"\n"}22: 3</span> — only the projects you care about</li>
                <li><b>One rank per line</b>: <span className="mono">5{"\n"}1{"\n"}3{"\n"}2</span> — line k = rank for project at index k-1; use 0 to skip</li>
              </ul>
            </div>
            <textarea
              value={prefText}
              onChange={(e) => setPrefText(e.target.value)}
              placeholder={`Examples:\n[6, 12, 32, 31, 30, 29, 25, 28, ...]\n\nor\n\n14: 1\n12: 2\n22: 3`}
              className="mono"
              style={{ flex: 1, minHeight: 240, fontSize: "0.82rem", lineHeight: 1.5 }}
            />
            {prefError && <p className="manual-error" style={{ marginTop: 8 }}>{prefError}</p>}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 16, gap: 12 }}>
              <span className="text-fog" style={{ fontSize: "0.78rem" }}>
                Updates your drag-list · projects you don't mention are appended in current order
              </span>
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn btn-secondary btn-sm" onClick={() => setPrefOpen(false)}>Cancel</button>
                <button className="btn btn-primary btn-sm" onClick={handlePrefImport}>Apply</button>
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