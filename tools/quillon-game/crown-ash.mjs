#!/usr/bin/env node
// Crown & Ash — a solo terminal mini in the Quillon celebration-art aesthetic.
//
// Each Claude Code instance plays its OWN game — no chain calls, no cross-
// instance coordination. Pick a faction. Manage your dynasty over 100 turns.
// Coronation, plagues, succession crises, jubilees, the final scene.
//
// Pure Node, single file, no deps. ANSI colors + box drawing + the same
// celebration-art idiom as tx_status (fireworks, rocket, gift, swap).
//
//   $ node crown-ash.mjs
//
// Controls: type the bracketed letter and press enter.

import readline from 'node:readline';
import { promisify } from 'node:util';

// ─── ANSI ─────────────────────────────────────────────────────────────
const ESC = '\x1b[';
const c = {
  reset:   ESC + '0m',
  bold:    ESC + '1m',
  dim:     ESC + '2m',
  italic:  ESC + '3m',
  // foregrounds
  red:     ESC + '38;5;196m',
  rose:    ESC + '38;5;197m',
  amber:   ESC + '38;5;214m',
  gold:    ESC + '38;5;220m',
  green:   ESC + '38;5;46m',
  emerald: ESC + '38;5;48m',
  cyan:    ESC + '38;5;51m',
  blue:    ESC + '38;5;39m',
  violet:  ESC + '38;5;135m',
  fuchsia: ESC + '38;5;201m',
  white:   ESC + '38;5;255m',
  slate:   ESC + '38;5;245m',
  dark:    ESC + '38;5;240m',
  bgCrown: ESC + '48;5;52m',
};
const w = (s) => process.stdout.write(s);
const clear = () => w(ESC + '2J' + ESC + 'H');
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ─── factions ─────────────────────────────────────────────────────────
const FACTIONS = [
  { id: 0, name: 'Frostspire',  emblem: '❄',  religion: 'Frost Spirits',  color: c.cyan,    motto: 'Where snow remembers.' },
  { id: 1, name: 'Sunmark',     emblem: '☀',  religion: 'The Solar Flame', color: c.amber,   motto: 'Light upon the throne.' },
  { id: 2, name: 'Ironwall',    emblem: '⚔',  religion: 'Forge of Hosts',  color: c.slate,   motto: 'A wall is a promise.' },
  { id: 3, name: 'Verdance',    emblem: '🌿', religion: 'The Green',       color: c.green,   motto: 'Roots outlast banners.' },
  { id: 4, name: 'Tideborn',    emblem: '🌊', religion: 'Salt & Star',     color: c.blue,    motto: 'The tide keeps its word.' },
  { id: 5, name: 'Stormcrown',  emblem: '⚡', religion: 'Stormfather',     color: c.violet,  motto: 'Lightning never rules twice.' },
  { id: 6, name: 'Embergate',   emblem: '🔥', religion: 'Last Ember',      color: c.rose,    motto: 'Even ash remembers fire.' },
];

const PROVINCE_NAMES = [
  'Greyhold', 'Westmarch', 'Auldfen', 'Briarcliff', 'Cinderhold', 'Mossdale',
  'Highreach', 'Saltlanding', 'Ravenmoor', 'Wolfswatch', 'Brightspire', 'Stonefell',
];

// ─── celebration art ──────────────────────────────────────────────────
function fireworks() {
  const lines = [
    `         ${c.gold}*${c.reset}    ${c.fuchsia}.${c.reset}      ${c.cyan}*${c.reset}`,
    `      ${c.amber}\\${c.reset} ${c.violet}|${c.reset} ${c.rose}/${c.reset}    ${c.gold}.${c.reset}  ${c.cyan}*${c.reset}`,
    `       ${c.amber}\\${c.reset}${c.bold}${c.gold}#${c.reset}${c.amber}/${c.reset}    ${c.fuchsia}*${c.reset}`,
    `        ${c.bold}${c.rose}V${c.reset}${c.amber}.${c.reset}`,
    `        ${c.dim}|${c.reset}`,
  ];
  return lines.join('\n');
}

function ashFall() {
  const lines = [
    `        ${c.dark}.${c.reset}    ${c.dark}.${c.reset}    ${c.dim}.${c.reset}`,
    `   ${c.slate}.${c.reset}     ${c.slate}.${c.reset}    ${c.dark}.${c.reset}    ${c.dim}.${c.reset}`,
    `       ${c.dark}.${c.reset}    ${c.slate}.${c.reset}    ${c.dark}.${c.reset}`,
    `   ${c.dim}—${c.reset} ${c.dark}ash falls quietly${c.reset} ${c.dim}—${c.reset}`,
  ];
  return lines.join('\n');
}

function crownBanner(faction) {
  const fc = faction.color;
  return [
    `   ${fc}${c.bold}╔═══════════════════════════════════════════╗${c.reset}`,
    `   ${fc}${c.bold}║${c.reset}   ${c.gold}${c.bold}♛${c.reset}  ${fc}${c.bold}House of ${faction.name}${c.reset}${' '.repeat(Math.max(0, 18 - faction.name.length))}${fc}${c.bold}${c.gold}♛${c.reset}  ${fc}${c.bold}║${c.reset}`,
    `   ${fc}${c.bold}║${c.reset}   ${c.italic}${c.slate}${faction.motto.padStart((43 + faction.motto.length) / 2 | 0).padEnd(37)}${c.reset}   ${fc}${c.bold}║${c.reset}`,
    `   ${fc}${c.bold}╚═══════════════════════════════════════════╝${c.reset}`,
  ].join('\n');
}

// ─── game state ───────────────────────────────────────────────────────
function makeProvince(name, baseProsperity = 1000) {
  return {
    name,
    population: 4000 + Math.floor(Math.random() * 6000),
    prosperity: baseProsperity + Math.floor(Math.random() * 500),
    food:   400 + Math.floor(Math.random() * 800),
    fortification: 1 + Math.floor(Math.random() * 3),
    unrest: Math.floor(Math.random() * 10),
    levy: 100 + Math.floor(Math.random() * 200),
  };
}

function newGame(faction) {
  return {
    faction,
    turn: 1,
    treasury: 2000,
    prestige: 100,
    age: 28,
    heir: null,
    realmName: faction.name,
    provinces: PROVINCE_NAMES.slice(0, 4).map(n => makeProvince(n)),
    log: [],
    events: [],
  };
}

// ─── render ───────────────────────────────────────────────────────────
function render(state) {
  clear();
  const f = state.faction;
  w(crownBanner(f) + '\n\n');

  const yearOfReign = state.turn;
  const eras = Math.floor((state.turn - 1) / 25);
  const eraNames = ['Sapling Years', 'Iron Years', 'Long Years', 'Final Years'];
  const eraName = eraNames[Math.min(eras, eraNames.length - 1)];

  w(`   ${c.dim}Turn ${state.turn} — Year ${yearOfReign} of the ${eraName}${c.reset}\n`);
  w(`   ${c.dim}Sovereign at age ${state.age}${state.heir ? `, heir: ${state.heir}` : ', no heir'}${c.reset}\n\n`);

  // Treasury / prestige bar
  const tBar = '█'.repeat(Math.min(30, Math.floor(state.treasury / 200)));
  const pBar = '█'.repeat(Math.min(30, Math.floor(state.prestige / 25)));
  w(`   ${c.gold}Treasury${c.reset}  ${c.gold}${tBar.padEnd(30)}${c.reset}  ${c.bold}${state.treasury}${c.reset} ${c.dim}gold${c.reset}\n`);
  w(`   ${c.violet}Prestige${c.reset}  ${c.violet}${pBar.padEnd(30)}${c.reset}  ${c.bold}${state.prestige}${c.reset}\n\n`);

  // Provinces
  w(`   ${c.bold}${c.white}Provinces (${state.provinces.length}):${c.reset}\n`);
  for (const p of state.provinces) {
    const unrestBar = p.unrest > 50 ? `${c.red}● HIGH${c.reset}` : p.unrest > 25 ? `${c.amber}● mid${c.reset}` : `${c.green}● calm${c.reset}`;
    const food = p.food < 200 ? `${c.red}${p.food}${c.reset}` : `${c.green}${p.food}${c.reset}`;
    w(`     ${f.color}${f.emblem}${c.reset}  ${c.bold}${p.name.padEnd(12)}${c.reset}  pop:${String(p.population).padStart(5)}  prosp:${String(p.prosperity).padStart(5)}  food:${food.padStart(13)}  fort:${'★'.repeat(p.fortification)}${' '.repeat(Math.max(0, 5 - p.fortification))}  ${unrestBar}\n`);
  }

  // Recent events
  if (state.events.length > 0) {
    w(`\n   ${c.bold}${c.amber}Tidings:${c.reset}\n`);
    for (const e of state.events.slice(-3)) {
      w(`     ${c.dim}—${c.reset} ${e}\n`);
    }
  }
  w('\n');
}

// ─── AI events (per turn) ─────────────────────────────────────────────
function tickEvents(state) {
  state.events = [];
  const rng = () => Math.random();

  // Tax collection
  let income = 0;
  for (const p of state.provinces) {
    const taxBase = Math.floor(p.prosperity * 0.05);
    const unrestPenalty = Math.floor(taxBase * (p.unrest / 100));
    income += taxBase - unrestPenalty;
    // Food consumption
    p.food = Math.max(0, p.food - Math.floor(p.population / 100));
    // Recovery
    p.unrest = Math.max(0, p.unrest - 1);
    p.population += Math.floor(p.prosperity / 100);
  }
  state.treasury += income;
  if (income > 0) state.events.push(`Tax collected: ${c.gold}+${income} gold${c.reset}.`);

  // Random events
  if (rng() < 0.12) {
    const p = state.provinces[Math.floor(rng() * state.provinces.length)];
    const lost = Math.floor(p.population * 0.15);
    p.population = Math.max(100, p.population - lost);
    p.unrest = Math.min(100, p.unrest + 20);
    state.events.push(`${c.red}Plague in ${p.name}.${c.reset} ${lost} dead, unrest rises.`);
    state.prestige = Math.max(0, state.prestige - 10);
  }
  if (rng() < 0.10) {
    const p = state.provinces[Math.floor(rng() * state.provinces.length)];
    p.food = Math.max(0, p.food - 300);
    p.unrest = Math.min(100, p.unrest + 15);
    state.events.push(`${c.amber}Famine in ${p.name}.${c.reset} Granaries fail.`);
  }
  if (rng() < 0.08 && !state.heir) {
    const heirNames = ['Yvain', 'Liesl', 'Tomas', 'Ildra', 'Cael', 'Mira', 'Bran'];
    state.heir = heirNames[Math.floor(rng() * heirNames.length)];
    state.events.push(`${c.fuchsia}An heir is born: ${state.heir}.${c.reset}`);
    state.prestige += 25;
  }
  if (rng() < 0.06) {
    state.prestige += 5;
    state.events.push(`${c.violet}A bard composes a song in your honour.${c.reset}`);
  }

  // Jubilee every 25 turns
  if (state.turn % 25 === 0) {
    state.prestige += 50;
    state.treasury += 1000;
    state.events.push(`${c.bold}${c.gold}A jubilee — the realm celebrates twenty-five years.${c.reset}`);
  }

  // Age
  state.age += 1;
  if (state.age > 65 && rng() < 0.10) {
    if (state.heir) {
      state.events.push(`${c.dim}You die at ${state.age}. ${state.heir} ascends.${c.reset}`);
      state.age = 22;
      state.heir = null;
      state.prestige = Math.floor(state.prestige * 0.7);
    } else {
      state.events.push(`${c.red}You die at ${state.age} without an heir.${c.reset}`);
      return { gameOver: true, reason: 'succession-crisis' };
    }
  }

  return { gameOver: false };
}

// ─── actions ──────────────────────────────────────────────────────────
const ACTIONS = [
  { key: 'B', name: 'Build a market',     cost: 500,  effect: (s, idx) => { s.provinces[idx].prosperity += 300; s.events.push(`Market built in ${c.bold}${s.provinces[idx].name}${c.reset}: +300 prosperity.`); } },
  { key: 'W', name: 'Strengthen the wall', cost: 400,  effect: (s, idx) => { s.provinces[idx].fortification += 1; s.events.push(`Wall raised in ${c.bold}${s.provinces[idx].name}${c.reset}: ${'★'.repeat(s.provinces[idx].fortification)}.`); } },
  { key: 'F', name: 'Feast the people',    cost: 300,  effect: (s, idx) => { s.provinces[idx].unrest = Math.max(0, s.provinces[idx].unrest - 30); s.prestige += 5; s.events.push(`Feast in ${c.bold}${s.provinces[idx].name}${c.reset}: unrest calmed.`); } },
  { key: 'L', name: 'Raise the levy',      cost: 200,  effect: (s, idx) => { s.provinces[idx].levy += 200; s.provinces[idx].unrest += 5; s.events.push(`Levy raised in ${c.bold}${s.provinces[idx].name}${c.reset}: +200 men, +5 unrest.`); } },
  { key: 'C', name: 'Tax cut',             cost: 0,    effect: (s, idx) => { s.provinces[idx].unrest = Math.max(0, s.provinces[idx].unrest - 15); s.treasury = Math.max(0, s.treasury - Math.floor(s.provinces[idx].prosperity * 0.03)); s.events.push(`Tax cut in ${c.bold}${s.provinces[idx].name}${c.reset}: people pleased.`); } },
  { key: 'P', name: 'Pray (no cost)',      cost: 0,    effect: (s)      => { s.prestige += 3; s.events.push(`A quiet prayer. +3 prestige.`); } },
];

function listActions(state) {
  const lines = [];
  for (const a of ACTIONS) {
    const aff = a.cost === 0 ? `${c.dim}free${c.reset}` : (state.treasury >= a.cost ? `${c.gold}${a.cost}g${c.reset}` : `${c.red}${a.cost}g${c.reset}`);
    lines.push(`     [${c.bold}${c.cyan}${a.key}${c.reset}] ${a.name.padEnd(22)} ${aff}`);
  }
  lines.push(`     [${c.bold}${c.cyan}E${c.reset}] End turn`);
  lines.push(`     [${c.bold}${c.red}Q${c.reset}] Abdicate (quit)`);
  return lines.join('\n');
}

// ─── input ────────────────────────────────────────────────────────────
function prompt(rl, q) {
  return new Promise(resolve => rl.question(q, ans => resolve(ans.trim())));
}

// ─── main loop ────────────────────────────────────────────────────────
async function pickFaction(rl) {
  clear();
  w(`   ${c.bold}${c.gold}♛  Crown & Ash — Choose Your House  ♛${c.reset}\n\n`);
  for (const f of FACTIONS) {
    w(`     [${c.bold}${c.cyan}${f.id}${c.reset}] ${f.color}${c.bold}${f.emblem}  ${f.name.padEnd(11)}${c.reset}  ${c.dim}— ${f.religion.padEnd(18)} — ${c.italic}${f.motto}${c.reset}\n`);
  }
  w('\n');
  while (true) {
    const ans = await prompt(rl, `   Pick a faction [0-6]: `);
    const id = parseInt(ans, 10);
    if (id >= 0 && id <= 6) return FACTIONS[id];
    w(`   ${c.red}Choose 0-6.${c.reset}\n`);
  }
}

async function coronationScene(faction) {
  clear();
  w(crownBanner(faction) + '\n\n');
  w(`   ${c.italic}${c.slate}You were not, until you signed your name with your hand.${c.reset}\n`);
  w(`   ${c.italic}${c.slate}You are now, because a chain says so.${c.reset}\n\n`);
  await sleep(900);
  w(fireworks() + '\n\n');
  await sleep(700);
  w(`   ${c.bold}${faction.color}A bell tolls. The realm is yours.${c.reset}\n\n`);
  await sleep(900);
  w(`   ${c.dim}Press enter to begin.${c.reset}`);
}

async function endScene(state, ok) {
  clear();
  w(crownBanner(state.faction) + '\n\n');
  if (ok) {
    w(fireworks() + '\n');
    w(`   ${c.bold}${c.gold}You ruled ${state.turn} turns. The chronicles remember.${c.reset}\n`);
    w(`   ${c.dim}Final treasury ${state.treasury}, prestige ${state.prestige}, age ${state.age}.${c.reset}\n`);
  } else {
    w(ashFall() + '\n\n');
    w(`   ${c.dim}${c.italic}The crown is set down. The next bell will toll for someone else.${c.reset}\n`);
    w(`   ${c.slate}You ruled ${state.turn} turns. Final treasury ${state.treasury}, prestige ${state.prestige}.${c.reset}\n`);
  }
  w('\n');
}

async function main() {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  const faction = await pickFaction(rl);
  await coronationScene(faction);
  await prompt(rl, '');

  const state = newGame(faction);

  while (true) {
    render(state);
    w(listActions(state) + '\n\n');

    if (state.treasury < 0) {
      state.events.push(`${c.red}The treasury runs dry. Lords desert.${c.reset}`);
      await sleep(800);
      await endScene(state, false);
      break;
    }

    const ans = (await prompt(rl, `   Your move: `)).toUpperCase();

    if (ans === 'Q') {
      await endScene(state, false);
      break;
    }

    if (ans === 'E') {
      const r = tickEvents(state);
      state.turn += 1;
      if (r.gameOver || state.turn > 100) {
        await endScene(state, !r.gameOver);
        break;
      }
      continue;
    }

    const a = ACTIONS.find(x => x.key === ans);
    if (!a) {
      w(`   ${c.red}Unknown move.${c.reset}\n`);
      await sleep(600);
      continue;
    }
    if (state.treasury < a.cost) {
      w(`   ${c.red}Treasury too low.${c.reset}\n`);
      await sleep(600);
      continue;
    }

    let provIdx = 0;
    if (a.name.includes('province') === false && a.name !== 'Pray (no cost)') {
      // ask which province
      w(`\n   Which province? (1-${state.provinces.length})\n`);
      state.provinces.forEach((p, i) => w(`     [${c.cyan}${i + 1}${c.reset}] ${p.name}\n`));
      const pAns = await prompt(rl, '   > ');
      provIdx = parseInt(pAns, 10) - 1;
      if (isNaN(provIdx) || provIdx < 0 || provIdx >= state.provinces.length) {
        w(`   ${c.red}Unknown province.${c.reset}\n`);
        await sleep(500);
        continue;
      }
    }

    state.treasury -= a.cost;
    a.effect(state, provIdx);
    await sleep(400);
  }

  rl.close();
}

main().catch(e => {
  console.error('Game error:', e);
  process.exit(1);
});
