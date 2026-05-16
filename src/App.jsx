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

// ==================== SIMULATION ====================
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

function randomPermutation(n, rng) {
  return rng.shuffle(Array.from({ length: n }, (_, i) => i + 1));
}

function clusteredPermutation(n, knownRankings, rng) {
  if (knownRankings.length === 0) return randomPermutation(n, rng);
  const marginals = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const ranking of knownRankings) {
    for (let p = 0; p < n; p++) {
      const r = ranking[p];
      if (r >= 1 && r <= n) marginals[p][r - 1] += 1;
    }
  }
  for (let p = 0; p < n; p++) {
    for (let r = 0; r < n; r++) marginals[p][r] += 0.5;
  }
  const projOrder = rng.shuffle(Array.from({ length: n }, (_, i) => i));
  const result = new Array(n).fill(0);
  const usedRanks = new Set();
  for (const p of projOrder) {
    const weights = marginals[p].map((w, r) => usedRanks.has(r + 1) ? 0 : w);
    const total = weights.reduce((a, b) => a + b, 0);
    if (total <= 0) {
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
      if (pickedR === -1) pickedR = n;
      result[p] = pickedR;
      usedRanks.add(pickedR);
    }
  }
  return result;
}

// ==================== HILL CLIMB ====================
function hillClimb(trainingWorlds, myTruePref, aiSet, maxIter = 300, restarts = 5, seed = null) {
  const n = myTruePref.length;
  const rngLocal = new SeededRandom(seed ?? 42);
  const honest = makeHonest(myTruePref);

  function avgUtility(ranking) {
    let total = 0;
    for (const opps of trainingWorlds) {
      total += utility(simulate(ranking, opps, "last"), myTruePref, aiSet);
    }
    return total / trainingWorlds.length;
  }

  let bestScore = avgUtility(honest);
  let bestRanking = [...honest];

  for (let restart = 0; restart < restarts; restart++) {
    let current = restart === 0 ? [...honest] : randomPermutation(n, rngLocal);
    let currentScore = avgUtility(current);
    for (let it = 0; it < maxIter; it++) {
      let bestSwap = null, bestSwapScore = currentScore;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          [current[i], current[j]] = [current[j], current[i]];
          const s = avgUtility(current);
          if (s < bestSwapScore) { bestSwapScore = s; bestSwap = [i, j]; }
          [current[i], current[j]] = [current[j], current[i]];
        }
      }
      if (!bestSwap) break;
      const [i, j] = bestSwap;
      [current[i], current[j]] = [current[j], current[i]];
      currentScore = bestSwapScore;
      if (currentScore <= -n) break;
    }
    if (currentScore < bestScore) { bestScore = currentScore; bestRanking = [...current]; }
  }
  return [bestRanking, bestScore];
}

// ==================== STRESS TEST HELPER ====================
// Runs the three stress scenarios comparing rankingA vs rankingB. Returns one
// row per scenario with all the metrics. Used by both robustnessCheck (for
// honest vs HC) and the custom-ranking section (for custom vs honest).
function stressTestRanking({
  rankingA,
  rankingB,
  honestRanking, // adversarial scenario uses this for unknown mirrors
  myTruePreferences,
  aiSet,
  knownRankings,
  missingCount,
  noiseTrials = 100,
  seed = null,
}) {
  const n = myTruePreferences.length;
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
      description: `Every unknown team submits the same ranking that you actually want most (your true preferences). Models "other students at the same school have similar tastes to me." Worst-case sidebar.`,
      sampleTrial: null,
    },
  ];

  const rng = new SeededRandom(seed ?? 42);
  const rows = [];

  for (const scenario of scenarios) {
    let aAi = 0, bAi = 0, aWins = 0, bWins = 0;
    const aRanks = [], bRanks = [];
    const aOutcomes = [], bOutcomes = [];
    for (let t = 0; t < noiseTrials; t++) {
      const totalOppsThisTrial = knownRankings.length + missingCount;
      const myRowForTrial = totalOppsThisTrial;
      let opps;
      if (scenario.key === "adversarial") {
        opps = [...knownRankings];
        for (let i = 0; i < missingCount; i++) opps.push([...honestRanking]);
      } else {
        opps = scenario.sampleTrial(rng);
      }
      const ap = simulate(rankingA, opps, myRowForTrial);
      const bp = simulate(rankingB, opps, myRowForTrial);
      const au = utility(ap, myTruePreferences, aiSet);
      const bu = utility(bp, myTruePreferences, aiSet);
      aOutcomes.push(ap); bOutcomes.push(bp);
      if (aiSet.has(ap)) aAi++; if (aiSet.has(bp)) bAi++;
      aRanks.push(myTruePreferences.indexOf(ap) + 1);
      bRanks.push(myTruePreferences.indexOf(bp) + 1);
      if (au < bu) aWins++; else if (bu < au) bWins++;
    }
    const medianOf = (arr) => {
      const s = [...arr].sort((a, b) => a - b);
      const m = s.length;
      if (m === 0) return 0;
      return m % 2 === 1 ? s[(m - 1) / 2] : (s[m / 2 - 1] + s[m / 2]) / 2;
    };
    rows.push({
      label: scenario.label,
      key: scenario.key,
      description: scenario.description,
      aAiPct: (aAi / noiseTrials) * 100,
      bAiPct: (bAi / noiseTrials) * 100,
      aMedian: medianOf(aRanks),
      bMedian: medianOf(bRanks),
      aWins, bWins,
      aOutcomes, bOutcomes,
    });
  }
  return rows;
}

// ==================== CUSTOM RANKING TEST ====================
// Stress-tests a user-supplied ranking against honest, using the same three
// scenarios as the main robustness check. Fast — no HC training needed.
function customRankingTest({
  customRanking,
  myTruePreferences,
  knownRankings,
  totalOpponents,
  aiProjectIndices,
  projectNames,
  noiseTrials = 100,
  seed = null,
}) {
  const n = myTruePreferences.length;
  const aiSet = new Set(aiProjectIndices);
  const missingCount = Math.max(0, totalOpponents - knownRankings.length);
  const honestRanking = makeHonest(myTruePreferences);
  const projNames = projectNames || Array.from({ length: n }, (_, i) => `Proj_${String(i).padStart(2, "0")}`);

  const rows = missingCount === 0
    ? null
    : stressTestRanking({
        rankingA: customRanking,
        rankingB: honestRanking,
        honestRanking,
        myTruePreferences,
        aiSet,
        knownRankings,
        missingCount,
        noiseTrials,
        seed,
      });

  // When there are no unknown teams, both outcomes are fully deterministic
  // (single Hungarian solve). Compute and surface them so the user sees the
  // actual outcome instead of just "deterministic".
  const deterministicOutcome = missingCount === 0
    ? {
        customProj: simulate(customRanking, knownRankings),
        honestProj: simulate(honestRanking, knownRankings),
      }
    : null;

  // Pool outcomes across the two realistic scenarios for the top-3 card
  // (excludes adversarial, same convention as the main result section).
  function topNModal(outcomes, k) {
    if (!outcomes || outcomes.length === 0) return [];
    const counts = new Map();
    for (const p of outcomes) counts.set(p, (counts.get(p) || 0) + 1);
    const total = outcomes.length;
    const entries = [];
    for (const [p, c] of counts.entries()) {
      entries.push({
        projIdx: p, count: c, total, pct: (c / total) * 100,
        truePref: myTruePreferences.indexOf(p),
      });
    }
    entries.sort((a, b) => (b.count !== a.count) ? b.count - a.count : a.truePref - b.truePref);
    return entries.slice(0, k);
  }

  const customOutcomesRealistic = rows
    ? rows.filter(r => r.key !== "adversarial").flatMap(r => r.aOutcomes)
    : [];
  const customTop3 = topNModal(customOutcomesRealistic, 3);

  // Worst outcome that occurred in adversarial trials (informational)
  const advRow = rows ? rows.find(r => r.key === "adversarial") : null;
  let customWorst = null;
  if (advRow && advRow.aOutcomes.length > 0) {
    let worstP = -1, worstTruePref = -1;
    for (const p of advRow.aOutcomes) {
      const tp = myTruePreferences.indexOf(p);
      if (tp > worstTruePref) { worstTruePref = tp; worstP = p; }
    }
    const count = advRow.aOutcomes.filter(p => p === worstP).length;
    customWorst = { projIdx: worstP, count, total: advRow.aOutcomes.length, pct: (count / advRow.aOutcomes.length) * 100 };
  }

  return {
    rows,
    customRanking,
    honestRanking,
    projectNames: projNames,
    aiSet,
    noiseTrials,
    missingCount,
    customTop3,
    customWorst,
    deterministicOutcome,
  };
}

// ==================== ROBUSTNESS CHECK ====================
function robustnessCheck({
  myTruePreferences,
  knownRankings,
  totalOpponents,
  aiProjectIndices,
  projectNames,
  noiseTrials = 100,
  seed = null,
  skipNoise = false,
  trainSize = 20,
}) {
  const n = myTruePreferences.length;
  const aiSet = new Set(aiProjectIndices);
  const missingCount = Math.max(0, totalOpponents - knownRankings.length);

  const projNames = projectNames || Array.from({ length: n }, (_, i) => `Proj_${String(i).padStart(2, "0")}`);
  const honestRanking = makeHonest(myTruePreferences);

  const rngClean = new SeededRandom(seed ?? 42);
  const cleanFill = [];
  for (let i = 0; i < missingCount; i++) cleanFill.push(randomPermutation(n, rngClean));
  const cleanOpponents = [...knownRankings, ...cleanFill];

  const hProjClean = simulate(honestRanking, cleanOpponents);
  const hScoreClean = utility(hProjClean, myTruePreferences, aiSet);

  const rngTrain = new SeededRandom((seed ?? 42) + 1);
  const trainingWorlds = [];
  if (missingCount === 0) {
    trainingWorlds.push([...knownRankings]);
  } else {
    const halfTrain = Math.ceil(trainSize / 2);
    for (let i = 0; i < halfTrain; i++) {
      const fill = [];
      for (let j = 0; j < missingCount; j++) fill.push(randomPermutation(n, rngTrain));
      trainingWorlds.push([...knownRankings, ...fill]);
    }
    for (let i = 0; i < trainSize - halfTrain; i++) {
      const fill = [];
      for (let j = 0; j < missingCount; j++) fill.push(clusteredPermutation(n, knownRankings, rngTrain));
      trainingWorlds.push([...knownRankings, ...fill]);
    }
  }

  const [hcRanking, hcScoreClean] = hillClimb(
    trainingWorlds, myTruePreferences, aiSet, 300, 3, seed
  );
  const hcProjClean = simulate(hcRanking, cleanOpponents);

  function modal(outcomes) {
    if (!outcomes || outcomes.length === 0) return null;
    const counts = new Map();
    for (const p of outcomes) counts.set(p, (counts.get(p) || 0) + 1);
    let bestP = -1, bestC = -1, bestTruePref = Infinity;
    for (const [p, c] of counts.entries()) {
      const tp = myTruePreferences.indexOf(p);
      if (c > bestC || (c === bestC && tp < bestTruePref)) {
        bestP = p; bestC = c; bestTruePref = tp;
      }
    }
    return { projIdx: bestP, count: bestC, total: outcomes.length, pct: (bestC / outcomes.length) * 100 };
  }

  function topNModal(outcomes, n) {
    if (!outcomes || outcomes.length === 0) return [];
    const counts = new Map();
    for (const p of outcomes) counts.set(p, (counts.get(p) || 0) + 1);
    const total = outcomes.length;
    const entries = [];
    for (const [p, c] of counts.entries()) {
      entries.push({
        projIdx: p, count: c, total, pct: (c / total) * 100,
        truePref: myTruePreferences.indexOf(p),
      });
    }
    entries.sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      return a.truePref - b.truePref;
    });
    return entries.slice(0, n);
  }

  function worst(outcomes) {
    if (!outcomes || outcomes.length === 0) return null;
    let worstP = -1, worstTruePref = -1;
    for (const p of outcomes) {
      const tp = myTruePreferences.indexOf(p);
      if (tp > worstTruePref) { worstTruePref = tp; worstP = p; }
    }
    const count = outcomes.filter(p => p === worstP).length;
    return { projIdx: worstP, count, total: outcomes.length, pct: (count / outcomes.length) * 100 };
  }

  if (hcScoreClean >= hScoreClean) {
    const deterministicTop3 = (projIdx) => [{ projIdx, count: 1, total: 1, pct: 100, truePref: myTruePreferences.indexOf(projIdx) }];
    const deterministicWorst = (projIdx) => ({ projIdx, countAdv: 1, totalAdv: 1, pctAdv: 100, countOverall: 1, totalOverall: 1, pctOverall: 100 });
    return {
      verdict: "SAFE", submit: "honest", honestRanking, hcRanking,
      reason: "HC found no improvement over honest. Nothing to risk.",
      knownCount: knownRankings.length, missingCount, projectNames: projNames,
      noiseResults: null, hProjClean, hcProjClean, noiseTrials, aiSet,
      hWorstCase: deterministicWorst(hProjClean),
      hcWorstCase: deterministicWorst(hcProjClean),
      hTop3: deterministicTop3(hProjClean),
      hcTop3: deterministicTop3(hcProjClean),
    };
  }
  if (skipNoise) {
    const deterministicTop3 = (projIdx) => [{ projIdx, count: 1, total: 1, pct: 100, truePref: myTruePreferences.indexOf(projIdx) }];
    const deterministicWorst = (projIdx) => ({ projIdx, countAdv: 1, totalAdv: 1, pctAdv: 100, countOverall: 1, totalOverall: 1, pctOverall: 100 });
    return {
      verdict: "SAFE", submit: "hc", honestRanking, hcRanking,
      reason: "Demo mode: HC improves on clean data. Skipped stress trials.",
      knownCount: knownRankings.length, missingCount, projectNames: projNames,
      noiseResults: null, hProjClean, hcProjClean, noiseTrials, aiSet,
      hWorstCase: deterministicWorst(hProjClean),
      hcWorstCase: deterministicWorst(hcProjClean),
      hTop3: deterministicTop3(hProjClean),
      hcTop3: deterministicTop3(hcProjClean),
    };
  }

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
      description: `Every unknown team submits the same ranking that you actually want most (your true preferences). Models "other students at the same school have similar tastes to me." Worst-case sidebar — not included in overall outcome aggregates because it represents a single extreme assumption, not the realistic distribution.`,
      sampleTrial: null,
    },
  ];

  const rng = new SeededRandom(seed ?? 42);
  const rows = [];

  for (const scenario of scenarios) {
    let hAi = 0, hcAi = 0, hcWins = 0, hWins = 0;
    const hPrefs = [], hcPrefs = [];
    const hRanks = [], hcRanks = [];
    const hOutcomes = [], hcOutcomes = [];
    for (let t = 0; t < noiseTrials; t++) {
      let hu, hcu, hp, hcp;
      const totalOppsThisTrial = knownRankings.length + missingCount;
      const myRowForTrial = totalOppsThisTrial;

      if (scenario.key === "adversarial") {
        const advOpps = [...knownRankings];
        for (let i = 0; i < missingCount; i++) advOpps.push([...honestRanking]);
        hp = simulate(honestRanking, advOpps, myRowForTrial);
        hu = utility(hp, myTruePreferences, aiSet);
        hcp = simulate(hcRanking, advOpps, myRowForTrial);
        hcu = utility(hcp, myTruePreferences, aiSet);
      } else {
        const opps = scenario.sampleTrial(rng);
        hp = simulate(honestRanking, opps, myRowForTrial);
        hu = utility(hp, myTruePreferences, aiSet);
        hcp = simulate(hcRanking, opps, myRowForTrial);
        hcu = utility(hcp, myTruePreferences, aiSet);
      }
      hOutcomes.push(hp);
      hcOutcomes.push(hcp);
      hPrefs.push(hu < 1000 ? hu : 99); if (aiSet.has(hp)) hAi++;
      hcPrefs.push(hcu < 1000 ? hcu : 99); if (aiSet.has(hcp)) hcAi++;
      hRanks.push(myTruePreferences.indexOf(hp) + 1);
      hcRanks.push(myTruePreferences.indexOf(hcp) + 1);
      if (hu < hcu) hWins++; else if (hcu < hu) hcWins++;
    }
    const sortedHRanks = [...hRanks].sort((a, b) => a - b);
    const sortedHcRanks = [...hcRanks].sort((a, b) => a - b);
    const medianOf = (sorted) => {
      const m = sorted.length;
      if (m === 0) return 0;
      return m % 2 === 1 ? sorted[(m - 1) / 2] : (sorted[m / 2 - 1] + sorted[m / 2]) / 2;
    };
    rows.push({
      label: scenario.label,
      key: scenario.key,
      description: scenario.description,
      hAiPct: (hAi / noiseTrials) * 100, hcAiPct: (hcAi / noiseTrials) * 100,
      hMean: hRanks.reduce((a, b) => a + b, 0) / hRanks.length,
      hcMean: hcRanks.reduce((a, b) => a + b, 0) / hcRanks.length,
      hMedian: medianOf(sortedHRanks),
      hcMedian: medianOf(sortedHcRanks),
      hcWins, hWins,
      hcCatPct: ((noiseTrials - hcAi) / noiseTrials) * 100,
      hCatPct: ((noiseTrials - hAi) / noiseTrials) * 100,
      hOutcomes, hcOutcomes,
    });
  }

  // ──────────────────────────────────────────────────────────────────────
  // Aggregate computation: EXCLUDE adversarial. Adversarial is shown as a
  // sidebar in the stress test table but does not contribute to the
  // "what's likely to happen" summary or to the verdict gating. Adversarial
  // is a single extreme assumption ("everyone else wants exactly what you
  // want"), not a representative sample of plausible worlds — pooling it
  // into the aggregates 1:1 with neutral and clustered would let one
  // worst-case assumption dominate the displayed probabilities.
  // ──────────────────────────────────────────────────────────────────────
  const realisticRows = rows.filter((r) => r.key !== "adversarial");
  const adversarialRow = rows.find((r) => r.key === "adversarial");

  const realisticHonestOutcomes = realisticRows.flatMap((r) => r.hOutcomes);
  const realisticHcOutcomes = realisticRows.flatMap((r) => r.hcOutcomes);

  // Top 3 outcomes computed from realistic (neutral + clustered) only.
  const hTop3 = topNModal(realisticHonestOutcomes, 3);
  const hcTop3 = topNModal(realisticHcOutcomes, 3);

  // Worst case: still derived from adversarial trials (the worst-case projects
  // ARE the adversarial outcomes), but `pctOverall` is computed against
  // realistic trials, not total. This lets the user see "if adversarial hits,
  // this is the bad project — and here's how often it shows up in realistic
  // worlds."
  function computeWorstCase(advOutcomes, realisticOutcomes) {
    const w = worst(advOutcomes);
    if (!w) return null;
    const realisticCount = realisticOutcomes.filter(p => p === w.projIdx).length;
    return {
      projIdx: w.projIdx,
      countAdv: w.count,
      totalAdv: w.total,
      pctAdv: w.pct,
      countOverall: realisticCount,
      totalOverall: realisticOutcomes.length,
      pctOverall: (realisticCount / realisticOutcomes.length) * 100,
    };
  }
  const hWorstCase = adversarialRow ? computeWorstCase(adversarialRow.hOutcomes, realisticHonestOutcomes) : null;
  const hcWorstCase = adversarialRow ? computeWorstCase(adversarialRow.hcOutcomes, realisticHcOutcomes) : null;

  const neutral = rows.find((r) => r.key === "neutral");
  const clustered = rows.find((r) => r.key === "clustered");
  const adversarial = adversarialRow;
  let verdict, submit, reason;

  // ──────────────────────────────────────────────────────────────────────
  // Verdict gating: neutral and clustered are the hard gates. Adversarial
  // is a SOFT WARNING — if HC catastrophically loses adversarial but wins
  // both realistic scenarios, the verdict is still SAFE/RISKY with a
  // caveat in the reason text. Adversarial alone can no longer flip a
  // SAFE verdict to UNSAFE.
  // ──────────────────────────────────────────────────────────────────────
  const neutralOK = neutral.hcWins >= noiseTrials * 0.45 || neutral.hcMedian <= neutral.hMedian;
  const clusteredOK = clustered.hcWins >= noiseTrials * 0.45 || clustered.hcMedian <= clustered.hMedian;
  const adversarialBadlyLoses = adversarial.hWins >= noiseTrials * 0.8;

  function buildAdvCaveat() {
    if (!adversarialBadlyLoses) return "";
    const worstName = projNames[hcWorstCase.projIdx];
    const advRank = myTruePreferences.indexOf(hcWorstCase.projIdx) + 1;
    return ` Note: in the worst-case scenario where every unknown team converges on your preferences, HC lands you on ${worstName} (your rank #${advRank}) in ${hcWorstCase.pctAdv.toFixed(0)}% of those trials. That scenario is a sidebar, not a typical outcome — but worth being aware of.`;
  }

  if (missingCount === 0) {
    verdict = "SAFE"; submit = "hc";
    reason = `All ${knownRankings.length} opponent rankings are known. HC outcome is deterministic.`;
  } else if (neutralOK && clusteredOK) {
    verdict = "SAFE"; submit = "hc";
    reason = `HC survives realistic scenarios. Neutral: ${neutral.hcWins}/${noiseTrials} wins. Clustered: ${clustered.hcWins}/${noiseTrials}.` + buildAdvCaveat();
  } else if (!neutralOK && !clusteredOK) {
    verdict = "UNSAFE"; submit = "honest";
    reason = `HC loses both realistic scenarios. Neutral: honest wins ${neutral.hWins}/${noiseTrials}. Clustered: honest wins ${clustered.hWins}/${noiseTrials}. The exploit doesn't survive normal unknown teams.`;
  } else {
    verdict = "RISKY"; submit = "honest";
    const losing = !neutralOK ? "neutral" : "clustered";
    const losingRow = !neutralOK ? neutral : clustered;
    reason = `HC wins one realistic scenario but loses ${losing} (honest wins ${losingRow.hWins}/${noiseTrials}). Edge is fragile.` + buildAdvCaveat();
  }

  return {
    verdict, submit, honestRanking, hcRanking, reason,
    knownCount: knownRankings.length, missingCount,
    projectNames: projNames, noiseResults: rows,
    hProjClean, hcProjClean, noiseTrials, aiSet,
    hWorstCase, hcWorstCase,
    hTop3, hcTop3,
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
  const [totalTeams, setTotalTeams] = useState(DEFAULT_TOTAL_TEAMS);
  const [jsonInput, setJsonInput] = useState("{\n  \n}");
  const [noiseTrials, setNoiseTrials] = useState(100);
  const [trainSize, setTrainSize] = useState(20);
  const [seed, setSeed] = useState(42);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  // Custom ranking tester state. customRanking is indexed by projIdx → submitted rank.
  // Starts as null (uninitialized); once initialized, it's a length-N array of ranks.
  const [customRanking, setCustomRanking] = useState(null);
  const [customResult, setCustomResult] = useState(null);
  const [customLoading, setCustomLoading] = useState(false);
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

  const parsePreferenceInput = (text) => {
    const trimmed = text.trim();
    if (!trimmed) return { error: "Paste at least one entry." };
    const n = projectNames.length;
    const projToRank = {};
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
      const lines = trimmed.split("\n").map((l) => l.trim()).filter(Boolean);
      const looksPaired = lines.every((l) => /^\s*\d+\s*[:\-—,]\s*\d+\s*$/.test(l)) ||
                           lines.every((l) => /^\s*\d+\s+\d+\s*$/.test(l));
      if (looksPaired) {
        for (const l of lines) {
          const m = l.match(/^\s*(\d+)\s*[:\-—,\s]+\s*(\d+)\s*$/);
          if (!m) return { error: `Could not parse "${l}" as "ID: rank".` };
          const externalId = Number(m[1]);
          const rank = Number(m[2]);
          let projIdx = -1;
          for (let i = 0; i < projectNames.length; i++) {
            const m2 = projectNames[i].match(/^(\d+)\s*:/);
            if (m2 && Number(m2[1]) === externalId) { projIdx = i; break; }
          }
          if (projIdx === -1) {
            if (externalId >= 1 && externalId <= n) projIdx = externalId - 1;
          }
          if (projIdx === -1) return { error: `No project found for ID ${externalId}.` };
          if (rank < 1 || rank > n) return { error: `Rank ${rank} out of range 1..${n}.` };
          projToRank[projIdx] = rank;
        }
      } else {
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
    const result = new Array(n).fill(-1);
    for (const [idxStr, r] of Object.entries(projToRank)) {
      result[r - 1] = Number(idxStr);
    }
    const usedProjects = new Set(Object.keys(projToRank).map(Number));
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
        trainSize,
      });
      setResult(check);
      setLoading(false);
    }, 50);
  };

  // Initialize the custom ranking from honest. Called when the user expands
  // the custom section for the first time, or clicks "Reset to Honest".
  const initCustomRanking = () => {
    const n = projectNames.length;
    const honest = new Array(n).fill(0);
    myTruePref.forEach((projIdx, rank) => { honest[projIdx] = rank + 1; });
    setCustomRanking(honest);
    setCustomResult(null);
  };

  // Update one project's rank. Allows duplicates intentionally — validation
  // is done at the time of running the stress test.
  const updateCustomRank = (projIdx, newRank) => {
    setCustomRanking((prev) => {
      if (!prev) return prev;
      const next = [...prev];
      next[projIdx] = newRank;
      return next;
    });
    setCustomResult(null); // any change invalidates prior result
  };

  // Validate custom ranking: must be a permutation of 1..n. Returns:
  //   { valid: true } or { valid: false, duplicates: [...], missing: [...], invalid: [...] }
  const validateCustomRanking = (ranking) => {
    if (!ranking) return { valid: false, duplicates: [], missing: [], invalid: [] };
    const n = projectNames.length;
    const seen = new Map(); // rank → [projIdx, ...]
    const invalid = []; // ranks out of 1..n
    for (let i = 0; i < n; i++) {
      const r = ranking[i];
      if (!Number.isInteger(r) || r < 1 || r > n) {
        invalid.push(i);
        continue;
      }
      if (!seen.has(r)) seen.set(r, []);
      seen.get(r).push(i);
    }
    const duplicates = [];
    for (const [r, idxs] of seen.entries()) {
      if (idxs.length > 1) duplicates.push({ rank: r, projIdxs: idxs });
    }
    const missing = [];
    for (let r = 1; r <= n; r++) if (!seen.has(r)) missing.push(r);
    return {
      valid: duplicates.length === 0 && missing.length === 0 && invalid.length === 0,
      duplicates, missing, invalid,
    };
  };

  const runCustomStressTest = () => {
    if (!customRanking) return;
    const val = validateCustomRanking(customRanking);
    if (!val.valid) return;
    const knownRankings = Object.values(publicRankings);
    const totalOpponents = Math.max(0, totalTeams - 1);
    setCustomLoading(true);
    setTimeout(() => {
      const check = customRankingTest({
        customRanking,
        myTruePreferences: myTruePref,
        knownRankings,
        totalOpponents,
        aiProjectIndices: Array.from(aiProjectIndices),
        projectNames,
        noiseTrials,
        seed,
      });
      setCustomResult(check);
      setCustomLoading(false);
    }, 50);
  };

  const copyOrderedList = async () => {
    if (!result) return;
    const ranking = result.submit === "hc" ? result.hcRanking : result.honestRanking;
    const indices = Array.from({ length: projectNames.length }, (_, i) => i)
      .sort((a, b) => ranking[a] - ranking[b]);
    const text = indices.map((idx, i) => `${i + 1}. ${projectNames[idx]}`).join("\n");
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
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
            <button className="btn btn-secondary btn-sm" onClick={() => { setPrefText(""); setPrefError(""); setPrefOpen(true); }}>Paste preferences</button>
            <button className="btn btn-secondary btn-sm" onClick={() => { setImportText(""); setImportError(""); setImportOpen(true); }}>Import projects</button>
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
              <div key={team} className="tracker-pill submitted"><span className="tracker-dot" />{team}</div>
            ))}
            {Array.from({ length: missingCount }, (_, i) => (
              <div key={`unknown-${i}`} className="tracker-pill waiting"><span className="tracker-dot" />unknown</div>
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
          <div>
            <p className="label" title="More worlds = more robust HC, but slower training. 20 ≈ 1 min, 30 ≈ 2 min.">HC Training Worlds</p>
            <input type="number" min={1} max={100} value={trainSize} onChange={(e) => setTrainSize(Math.max(1, Number(e.target.value) || 1))} />
          </div>
          <div><p className="label">Random Seed</p><input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></div>
        </div>
        <div className="mode-toggle-row">
          <button className={`btn btn-sm ${demoMode ? "btn-primary" : "btn-secondary"}`} onClick={() => setDemoMode(true)}>Demo Mode</button>
          <button className={`btn btn-sm ${!demoMode ? "btn-primary" : "btn-secondary"}`} onClick={() => setDemoMode(false)}>Real Data</button>
          <span className="text-fog" style={{ fontSize: "0.85rem" }}>{demoMode ? "Skips stress trials — instant results for testing" : "Full robustness check with 3-scenario stress test"}</span>
        </div>
      </section>

      <button className="btn btn-primary w-full mt-4" onClick={runAnalysis} disabled={loading}>
        {loading ? `Training HC across ${trainSize} worlds…` : demoMode ? "Run Quick Check" : "Run Robustness Check"}
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
            <OutcomeCard
              title="Honest Outcome"
              top3={result.hTop3}
              worst={result.hWorstCase}
              projectNames={result.projectNames}
              aiSet={result.aiSet}
              myTruePref={myTruePref}
            />
            <OutcomeCard
              title="HC Outcome"
              top3={result.hcTop3}
              worst={result.hcWorstCase}
              projectNames={result.projectNames}
              aiSet={result.aiSet}
              myTruePref={myTruePref}
            />
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
                const indices = Array.from({ length: projectNames.length }, (_, i) => i).sort((a, b) => ranking[a] - ranking[b]);
                return indices.map((projIdx) => (
                  <li key={projIdx} style={{ marginBottom: 4, breakInside: "avoid" }}>
                    {projectNames[projIdx]}
                    {result.aiSet.has(projIdx) && <span className="ai-badge" style={{ marginLeft: 8 }}>AI</span>}
                  </li>
                ));
              })()}
            </ol>
          </div>

          {result.submit === "honest" && JSON.stringify(result.honestRanking) !== JSON.stringify(result.hcRanking) && (
            <div className="card mt-4" style={{ borderLeft: "3px solid var(--ember)" }}>
              <div className="flex justify-between items-center" style={{ marginBottom: 4 }}>
                <h3 style={{ fontFamily: "Newsreader, serif", margin: 0 }}>Hill Climb's alternative ranking</h3>
                <span className="text-fog" style={{ fontSize: "0.8rem" }}>(not recommended)</span>
              </div>
              <p className="label" style={{ marginTop: 0, marginBottom: 12 }}>What HC would have submitted instead — shown so you can compare.</p>
              <ol style={{ paddingLeft: 24, margin: 0, columnCount: 2, columnGap: 24, fontFamily: "Inter Tight, sans-serif", fontSize: "0.9rem" }}>
                {(() => {
                  const ranking = result.hcRanking;
                  const indices = Array.from({ length: projectNames.length }, (_, i) => i).sort((a, b) => ranking[a] - ranking[b]);
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
                {result.knownCount} known ranking{result.knownCount === 1 ? "" : "s"} frozen. Only the {result.missingCount} unknown team{result.missingCount === 1 ? "" : "s"} {result.missingCount === 1 ? "is" : "are"} sampled per trial. Adversarial scenario is informational only — it does not affect the verdict or the Top 3 outcomes.
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
                      <th>HC Median</th>
                      <th>Hon Median</th>
                      <th>HC Mean</th>
                      <th>Hon Mean</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.noiseResults.map((row, i) => (
                      <tr key={i} style={row.key === "adversarial" ? { opacity: 0.7 } : {}}>
                        <td style={{ fontFamily: "Inter Tight, sans-serif", verticalAlign: "top", maxWidth: 280 }}>
                          <div style={{ fontWeight: 600 }}>{row.label}{row.key === "adversarial" && <span style={{ marginLeft: 8, fontSize: "0.7rem", color: "var(--fog)", fontWeight: 400 }}>sidebar</span>}</div>
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
                        <td style={{ verticalAlign: "top" }} className={row.hcMedian < row.hMedian ? "text-olive" : row.hcMedian > row.hMedian ? "text-ember" : ""}>{row.hcMedian.toFixed(1)}</td>
                        <td style={{ verticalAlign: "top" }}>{row.hMedian.toFixed(1)}</td>
                        <td style={{ verticalAlign: "top" }} className={row.hcMean < row.hMean ? "text-olive" : row.hcMean > row.hMean ? "text-ember" : ""}>{row.hcMean.toFixed(1)}</td>
                        <td style={{ verticalAlign: "top" }}>{row.hMean.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-fog" style={{ fontSize: "0.78rem", marginTop: 12, marginBottom: 0, lineHeight: 1.5 }}>
                <b>Median</b> = middle outcome rank across trials. Robust to extreme outliers; tells you "the typical outcome."<br />
                <b>Mean</b> = average outcome rank. Sensitive to tails; lower mean means HC's distribution is more concentrated near your top picks.<br />
                When <b>Median {"<"} Mean</b> (gap pulling right), the distribution has a heavy right tail — HC mostly lands well but occasionally lands badly. When both agree, the outcome is consistent.<br />
                <b>Wins</b> = trials where that ranking gave a strictly better utility (rank + small AI bonus). Ties don't count for either side.<br />
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

      {/* ==================== CUSTOM RANKING TESTER ==================== */}
      <section className="card mt-4">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8, gap: 8, flexWrap: "wrap" }}>
          <div>
            <h2 style={{ margin: 0 }}>Try Your Own Ranking</h2>
            <p className="label" style={{ marginTop: 4, marginBottom: 0 }}>
              Enter any custom ranking and stress-test it against the same three scenarios. Useful for exploring "what if I submitted X?"
            </p>
          </div>
          {customRanking ? (
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn btn-secondary btn-sm" onClick={initCustomRanking}>Reset to Honest</button>
              <button className="btn btn-secondary btn-sm" onClick={() => { setCustomRanking(null); setCustomResult(null); }}>Hide</button>
            </div>
          ) : (
            <button className="btn btn-primary btn-sm" onClick={initCustomRanking}>Start Building</button>
          )}
        </div>

        {customRanking && (() => {
          const val = validateCustomRanking(customRanking);
          // Build a map: projIdx → "duplicate" | "missing-flag" for tile borders.
          const dupeProjIdxs = new Set();
          for (const d of val.duplicates) for (const pi of d.projIdxs) dupeProjIdxs.add(pi);
          const invalidProjIdxs = new Set(val.invalid);
          const totalOpp = Math.max(0, totalTeams - 1);
          const enteredOpp = Object.keys(publicRankings).length;
          const tooManyKnown = enteredOpp > totalOpp;
          return (
            <>
              {/* status pill */}
              <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 16, marginBottom: 12, flexWrap: "wrap" }}>
                {val.valid ? (
                  <span className="json-status-pill ok"><span className="json-status-dot" />valid permutation</span>
                ) : (
                  <>
                    <span className="json-status-pill err"><span className="json-status-dot" />invalid — fix conflicts below</span>
                    {val.duplicates.length > 0 && (
                      <span className="text-fog mono" style={{ fontSize: "0.78rem" }}>
                        duplicate rank{val.duplicates.length === 1 ? "" : "s"}: {val.duplicates.map(d => d.rank).sort((a, b) => a - b).join(", ")}
                      </span>
                    )}
                    {val.missing.length > 0 && val.missing.length <= 8 && (
                      <span className="text-fog mono" style={{ fontSize: "0.78rem" }}>
                        missing rank{val.missing.length === 1 ? "" : "s"}: {val.missing.join(", ")}
                      </span>
                    )}
                    {val.missing.length > 8 && (
                      <span className="text-fog mono" style={{ fontSize: "0.78rem" }}>
                        {val.missing.length} ranks missing
                      </span>
                    )}
                  </>
                )}
                <button
                  className="btn btn-primary btn-sm"
                  style={{ marginLeft: "auto" }}
                  onClick={runCustomStressTest}
                  disabled={!val.valid || customLoading || tooManyKnown}
                  title={!val.valid ? "Fix duplicates/missing ranks first" : tooManyKnown ? "Too many known opponents for current team count" : ""}
                >
                  {customLoading ? "Running…" : "Run Stress Test"}
                </button>
              </div>

              {/* Tile grid */}
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
                gap: 8,
                marginTop: 8,
              }}>
                {projectNames.map((name, projIdx) => {
                  const r = customRanking[projIdx];
                  const isDupe = dupeProjIdxs.has(projIdx);
                  const isInvalid = invalidProjIdxs.has(projIdx);
                  const isAi = aiProjectIndices.has(projIdx);
                  return (
                    <div
                      key={projIdx}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        padding: "8px 10px",
                        background: "var(--paper)",
                        border: `1px solid ${isDupe || isInvalid ? "var(--ember)" : "var(--rule)"}`,
                        borderRadius: 8,
                        fontSize: "0.85rem",
                        lineHeight: 1.3,
                      }}
                    >
                      <input
                        type="number"
                        min={1}
                        max={projectNames.length}
                        value={Number.isInteger(r) && r >= 1 ? r : ""}
                        onChange={(e) => {
                          const v = e.target.value;
                          updateCustomRank(projIdx, v === "" ? 0 : Math.max(1, Math.min(projectNames.length, Number(v) || 0)));
                        }}
                        style={{
                          width: 52,
                          padding: "4px 6px",
                          fontSize: "0.85rem",
                          fontFamily: "JetBrains Mono, monospace",
                          fontWeight: 600,
                          textAlign: "center",
                          border: `1px solid ${isDupe || isInvalid ? "var(--ember)" : "var(--rule)"}`,
                          borderRadius: 4,
                          background: isDupe || isInvalid ? "rgba(207, 99, 71, 0.08)" : "var(--bone)",
                          color: isDupe || isInvalid ? "var(--ember)" : "inherit",
                          flexShrink: 0,
                        }}
                      />
                      <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={name}>
                        {name}
                      </span>
                      {isAi && <span className="ai-badge">AI</span>}
                    </div>
                  );
                })}
              </div>

              {/* Results */}
              {customResult && customResult.rows && (
                <div style={{ marginTop: 24 }}>
                  <h3 style={{ fontFamily: "Newsreader, serif", margin: "0 0 4px" }}>Custom Ranking Results</h3>
                  <p className="label" style={{ margin: "0 0 12px" }}>
                    Your custom ranking vs honest, same three scenarios as the main robustness check.
                  </p>

                  {/* Top-3 outcome card for custom */}
                  <div style={{ marginBottom: 16 }}>
                    <div className="result-card">
                      <h4>Custom Ranking Outcome</h4>
                      <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--rule)" }}>
                        <div style={{ fontSize: "0.7rem", fontWeight: 600, color: "var(--fog)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
                          Top 3 most likely outcomes (realistic scenarios pooled)
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                          {customResult.customTop3.map((entry, i) => (
                            <div key={i} style={{ display: "grid", gridTemplateColumns: "auto 1fr auto", alignItems: "baseline", gap: 10, fontSize: "0.88rem", lineHeight: 1.35 }}>
                              <span className="mono" style={{ color: "var(--fog)", fontSize: "0.78rem", minWidth: 14 }}>{i + 1}.</span>
                              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={projectNames[entry.projIdx]}>
                                {projectNames[entry.projIdx]}
                                <span className="mono" style={{ color: "var(--fog)", marginLeft: 8, fontSize: "0.74rem" }}>
                                  #{entry.truePref + 1}{aiProjectIndices.has(entry.projIdx) ? " · AI" : ""}
                                </span>
                              </span>
                              <span className="mono" style={{ fontWeight: 600, fontSize: "0.88rem", whiteSpace: "nowrap" }}>
                                {entry.pct.toFixed(0)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                      {customResult.customWorst && (
                        <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--rule)" }}>
                          <div style={{ fontSize: "0.7rem", fontWeight: 600, color: "var(--fog)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
                            Worst outcome under adversarial
                          </div>
                          <div style={{ display: "grid", gridTemplateColumns: "1fr auto", alignItems: "baseline", gap: 10, fontSize: "0.88rem", lineHeight: 1.35 }}>
                            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={projectNames[customResult.customWorst.projIdx]}>
                              {projectNames[customResult.customWorst.projIdx]}
                              <span className="mono" style={{ color: "var(--fog)", marginLeft: 8, fontSize: "0.74rem" }}>
                                #{myTruePref.indexOf(customResult.customWorst.projIdx) + 1}{aiProjectIndices.has(customResult.customWorst.projIdx) ? " · AI" : ""}
                              </span>
                            </span>
                            <span className="mono" style={{ fontWeight: 600, fontSize: "0.88rem" }}>
                              {customResult.customWorst.pct.toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Stress test table */}
                  <div style={{ overflowX: "auto" }}>
                    <table className="data-table mono">
                      <thead>
                        <tr>
                          <th>Scenario</th>
                          <th>Hon AI%</th>
                          <th>Custom AI%</th>
                          <th>Custom Wins</th>
                          <th>Honest Wins</th>
                          <th>Custom Median</th>
                          <th>Honest Median</th>
                        </tr>
                      </thead>
                      <tbody>
                        {customResult.rows.map((row, i) => (
                          <tr key={i}>
                            <td style={{ fontFamily: "Inter Tight, sans-serif", verticalAlign: "top", maxWidth: 280 }}>
                              <div style={{ fontWeight: 600 }}>{row.label}</div>
                              {row.description && (
                                <div style={{ fontSize: "0.72rem", color: "var(--fog)", marginTop: 4, lineHeight: 1.4, fontFamily: "Inter Tight, sans-serif", whiteSpace: "normal" }}>
                                  {row.description}
                                </div>
                              )}
                            </td>
                            <td style={{ verticalAlign: "top" }}>{row.bAiPct.toFixed(0)}%</td>
                            <td style={{ verticalAlign: "top" }}>{row.aAiPct.toFixed(0)}%</td>
                            <td style={{ verticalAlign: "top" }} className={row.aWins > row.bWins ? "text-olive" : ""}>{row.aWins}/{customResult.noiseTrials}</td>
                            <td style={{ verticalAlign: "top" }} className={row.bWins > row.aWins ? "text-ember" : ""}>{row.bWins}/{customResult.noiseTrials}</td>
                            <td style={{ verticalAlign: "top" }}>{row.aMedian.toFixed(1)}</td>
                            <td style={{ verticalAlign: "top" }}>{row.bMedian.toFixed(1)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {customResult && !customResult.rows && customResult.deterministicOutcome && (() => {
                const det = customResult.deterministicOutcome;
                const customRank = myTruePref.indexOf(det.customProj) + 1;
                const honestRank = myTruePref.indexOf(det.honestProj) + 1;
                const same = det.customProj === det.honestProj;
                return (
                  <div style={{ marginTop: 24 }}>
                    <h3 style={{ fontFamily: "Newsreader, serif", margin: "0 0 4px" }}>Deterministic Outcome</h3>
                    <p className="label" style={{ margin: "0 0 12px" }}>
                      All {Object.keys(publicRankings).length} opponent rankings are known. No stress test needed — there's only one possible outcome per ranking.
                    </p>
                    <div className="result-grid">
                      <div className="result-card">
                        <h4>Custom Ranking Outcome</h4>
                        <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--rule)", fontSize: "0.95rem", lineHeight: 1.4 }}>
                          <div style={{ fontWeight: 600 }}>{projectNames[det.customProj]}</div>
                          <div className="mono" style={{ color: "var(--fog)", fontSize: "0.8rem", marginTop: 4 }}>
                            your true #{customRank}{aiProjectIndices.has(det.customProj) ? " · AI" : ""}
                          </div>
                        </div>
                      </div>
                      <div className="result-card">
                        <h4>Honest Ranking Outcome</h4>
                        <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--rule)", fontSize: "0.95rem", lineHeight: 1.4 }}>
                          <div style={{ fontWeight: 600 }}>{projectNames[det.honestProj]}</div>
                          <div className="mono" style={{ color: "var(--fog)", fontSize: "0.8rem", marginTop: 4 }}>
                            your true #{honestRank}{aiProjectIndices.has(det.honestProj) ? " · AI" : ""}
                          </div>
                        </div>
                      </div>
                    </div>
                    <div style={{ marginTop: 12, padding: "10px 14px", background: "var(--bone)", borderRadius: 8, fontSize: "0.85rem" }}>
                      {same
                        ? <>Custom and honest give the <b>same outcome</b>. No advantage either way.</>
                        : customRank < honestRank
                          ? <>Custom does <b className="text-olive">better</b> than honest by {honestRank - customRank} preference position{honestRank - customRank === 1 ? "" : "s"}.</>
                          : <>Custom does <b className="text-ember">worse</b> than honest by {customRank - honestRank} preference position{customRank - honestRank === 1 ? "" : "s"}.</>
                      }
                    </div>
                  </div>
                );
              })()}
            </>
          );
        })()}
      </section>

      {importOpen && (
        <div onClick={() => setImportOpen(false)} style={{ position: "fixed", inset: 0, background: "rgba(14,17,22,0.45)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, padding: 20 }}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "var(--paper)", border: "1px solid var(--rule)", borderRadius: 12, padding: 24, maxWidth: 640, width: "100%", maxHeight: "85vh", display: "flex", flexDirection: "column", boxShadow: "0 20px 60px rgba(14,17,22,0.25)" }}>
            <h2 style={{ marginBottom: 8 }}>Import Projects</h2>
            <p className="label" style={{ marginTop: 0, marginBottom: 12 }}>Paste one project per line. Format: <span className="mono">ID: Name</span> or just the name. Rows will be sorted by ID if all have one.</p>
            <textarea value={importText} onChange={(e) => setImportText(e.target.value)} placeholder={`1: Type Safe LLM Library\n2: Advising Chat Bot\n3: Moodle Date Editing\n...`} className="mono" style={{ flex: 1, minHeight: 280, fontSize: "0.82rem", lineHeight: 1.5 }} />
            {importError && <p className="manual-error" style={{ marginTop: 8 }}>{importError}</p>}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 16, gap: 12 }}>
              <span className="text-fog" style={{ fontSize: "0.78rem" }}>Replaces all project names · resets AI flags · keeps your preference order where possible</span>
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
            <p className="label" style={{ marginTop: 0, marginBottom: 12 }}>Same semantic as opponent rankings: <span className="mono">arr[i] = rank you assign to project i</span>. Partial input is OK — unranked projects keep their current order at the end.</p>
            <div style={{ fontSize: "0.78rem", color: "var(--fog)", marginBottom: 12, lineHeight: 1.6 }}>
              Three accepted formats:
              <ul style={{ margin: "4px 0 0 18px", padding: 0 }}>
                <li><b>JSON array</b>: <span className="mono">[6, 12, 32, 31, ...]</span> — same as opponent ranking format</li>
                <li><b>ID: rank pairs</b>: <span className="mono">14: 1{"\n"}12: 2{"\n"}22: 3</span> — only the projects you care about</li>
                <li><b>One rank per line</b>: <span className="mono">5{"\n"}1{"\n"}3{"\n"}2</span> — line k = rank for project at index k-1; use 0 to skip</li>
              </ul>
            </div>
            <textarea value={prefText} onChange={(e) => setPrefText(e.target.value)} placeholder={`Examples:\n[6, 12, 32, 31, 30, 29, 25, 28, ...]\n\nor\n\n14: 1\n12: 2\n22: 3`} className="mono" style={{ flex: 1, minHeight: 240, fontSize: "0.82rem", lineHeight: 1.5 }} />
            {prefError && <p className="manual-error" style={{ marginTop: 8 }}>{prefError}</p>}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 16, gap: 12 }}>
              <span className="text-fog" style={{ fontSize: "0.78rem" }}>Updates your drag-list · projects you don't mention are appended in current order</span>
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

// ==================== OUTCOME CARD ====================
function OutcomeCard({ title, top3, worst, projectNames, aiSet, myTruePref }) {
  return (
    <div className="result-card">
      <h4>{title}</h4>

      <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--rule)" }}>
        <div style={{ fontSize: "0.7rem", fontWeight: 600, color: "var(--fog)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          Top 3 most likely outcomes
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {top3.map((entry, i) => (
            <div
              key={i}
              style={{
                display: "grid",
                gridTemplateColumns: "auto 1fr auto",
                alignItems: "baseline",
                gap: 10,
                fontSize: "0.88rem",
                lineHeight: 1.35,
              }}
            >
              <span className="mono" style={{ color: "var(--fog)", fontSize: "0.78rem", minWidth: 14 }}>{i + 1}.</span>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={projectNames[entry.projIdx]}>
                {projectNames[entry.projIdx]}
                <span className="mono" style={{ color: "var(--fog)", marginLeft: 8, fontSize: "0.74rem" }}>
                  #{entry.truePref + 1}{aiSet.has(entry.projIdx) ? " · AI" : ""}
                </span>
              </span>
              <span className="mono" style={{ fontWeight: 600, fontSize: "0.88rem", whiteSpace: "nowrap" }}>
                {entry.pct.toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--rule)" }}>
        <div style={{ fontSize: "0.7rem", fontWeight: 600, color: "var(--fog)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          Worst-case sidebar (adversarial)
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr auto",
            alignItems: "baseline",
            gap: 10,
            fontSize: "0.88rem",
            lineHeight: 1.35,
          }}
        >
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={projectNames[worst.projIdx]}>
            {projectNames[worst.projIdx]}
            <span className="mono" style={{ color: "var(--fog)", marginLeft: 8, fontSize: "0.74rem" }}>
              #{myTruePref.indexOf(worst.projIdx) + 1}{aiSet.has(worst.projIdx) ? " · AI" : ""}
            </span>
          </span>
          <span className="mono" style={{ fontWeight: 600, fontSize: "0.88rem", whiteSpace: "nowrap" }}>
            {worst.pctAdv.toFixed(0)}%
          </span>
        </div>
        <div className="mono" style={{ color: "var(--fog)", fontSize: "0.72rem", marginTop: 4 }}>
          {worst.pctAdv.toFixed(0)}% of adversarial trials · {worst.pctOverall.toFixed(0)}% in realistic worlds
        </div>
      </div>
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