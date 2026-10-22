import { memo, useMemo, useState } from "react";
import {
  HORIZONS, LEVELS, LEVEL_RANK, clock, fmt, level, pct, sexLabel, type BedInfo, type BedSnap, type Horizon, type Level,
  type ModelInfo, type ModelName,
} from "../api";
import type { Buffer } from "../useWard";
import { StatusPill } from "./Panels";

const SPARK_S = 600;

/** MAP of the last 10 min (2 s samples), 65 mmHg line; below 65 drawn in the critical colour. */
function Spark({ buf, now }: { buf: Buffer | undefined; now: number }) {
  const W = 600, H = 60, lo = 40, hi = 120;
  const y = (v: number) => H - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * H;
  const { d, low } = useMemo(() => {
    let d = "", low = "", pen = false, penL = false;
    if (buf) for (let i = 0; i < buf.t.length; i++) {
      const t = buf.t[i], v = buf.map[i];
      if (t < now - SPARK_S) continue;
      if (v == null) { pen = penL = false; continue; }
      const xx = (W * (t - (now - SPARK_S)) / SPARK_S).toFixed(1), yy = y(v).toFixed(1);
      d += `${pen ? "L" : "M"}${xx},${yy}`; pen = true;
      if (v < 65) { low += `${penL ? "L" : "M"}${xx},${yy}`; penL = true; } else penL = false;
    }
    return { d, low };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [buf, buf?.t.length, now]);
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden>
      <line x1={0} x2={W} y1={y(65)} y2={y(65)} className="spark-65" />
      <path d={d} className="spark-line" />
      {low && <path d={low} className="spark-low" />}
    </svg>
  );
}

function MiniBars({ risk, thr, horizon }: { risk: number[] | null; thr: number[]; horizon: Horizon }) {
  return (
    <div className="minibars" aria-hidden>
      {HORIZONS.map((h, k) => {
        const p = risk?.[k] ?? 0, over = risk != null && p >= thr[k];
        return (
          <div key={h} className={`mb${h === horizon ? " sel" : ""}`} title={`${h / 60}′: ${risk ? pct(p) : "–"}%`}>
            <div className="mb-track"><i style={{ height: `${Math.max(p * 100, 4)}%`, background: over ? "var(--serious)" : "var(--series-1)" }} /></div>
            <span>{h / 60}′</span>
          </div>
        );
      })}
    </div>
  );
}

interface TileProps {
  info: BedInfo; snap: BedSnap; buf: Buffer | undefined; model: ModelName; horizon: Horizon; thr: number[];
  onOpen: (bed: number) => void;
}

const BedTile = memo(function BedTile({ info, snap, buf, model, horizon, thr, onOpen }: TileProps) {
  const k = HORIZONS.indexOf(horizon);
  const r = snap.risk[model];
  const p = r ? r[k] : null;
  const lv: Level = level(p, thr[k], snap.alarm[model][k]);
  const v = snap.vitals;
  const lowMap = v.map != null && v.map < 65;
  return (
    <button className={`tile lv-${lv}`} onClick={() => onOpen(info.bed)} aria-label={`Giường ${info.bed}, ${LEVELS[lv].label}`}>
      <div className="tile-head">
        <b>Giường {String(info.bed).padStart(2, "0")}</b>
        {snap.in_event && <span className="chip-ev">MAP &lt; 65</span>}
        <StatusPill lv={lv} compact />
      </div>
      <div className="tile-sub">
        {sexLabel(info.sex)}, {fmt(info.age)} tuổi · ASA {info.asa ?? "–"}{info.emergency ? " · cấp cứu" : ""}
        <br />{info.department} · {snap.phase === "pre_incision" ? "trước rạch da" : "trong mổ"} · <span className="num">{clock(snap.elapsed_s).slice(0, 5)}</span>
      </div>
      <div className="tile-main">
        <div>
          <div className="tile-risk num" style={{ color: lv === "na" ? "var(--muted)" : LEVELS[lv].color }}>
            {p == null ? "–" : pct(p)}<small>{p == null ? "" : "%"}</small>
          </div>
          <div className="tile-cap">nguy cơ {horizon / 60}′</div>
        </div>
        <div className="tile-map">
          <div className={`num${lowMap ? " low" : ""}`}>{fmt(v.map)}</div>
          <div className="tile-cap">MAP mmHg</div>
        </div>
      </div>
      <Spark buf={buf} now={snap.t} />
      <div className="tile-foot">
        <span className="vt"><em>HR</em><b className="num">{fmt(v.hr)}</b></span>
        <span className="vt"><em>SpO₂</em><b className="num">{fmt(v.spo2)}</b></span>
        <MiniBars risk={r} thr={thr} horizon={horizon} />
      </div>
    </button>
  );
});

export function WardGrid({ beds, snaps, buffers, models, model, horizon, onOpen }: {
  beds: BedInfo[]; snaps: Map<number, BedSnap>; buffers: Map<number, Buffer>; models: ModelInfo[];
  model: ModelName; horizon: Horizon; onOpen: (bed: number) => void;
}) {
  const [sort, setSort] = useState<"bed" | "risk">("risk");
  const mi = models.find((m) => m.name === model);
  const thr = useMemo(() => HORIZONS.map((h) => mi?.horizons[String(h)].threshold ?? 0.5), [mi]);
  const k = HORIZONS.indexOf(horizon);
  const rows = beds.filter((b) => snaps.has(b.bed)).map((b) => {
    const s = snaps.get(b.bed)!, p = s.risk[model]?.[k] ?? null;
    return { b, s, p, lv: level(p, thr[k], s.alarm[model][k]) };
  });
  if (sort === "risk") rows.sort((a, c) => LEVEL_RANK[c.lv] - LEVEL_RANK[a.lv] || (c.p ?? -1) - (a.p ?? -1) || a.b.bed - c.b.bed);
  const count = (l: Level) => rows.filter((r) => r.lv === l).length;
  return (
    <>
      <div className="ward-bar">
        <div className="counts">
          {(["alarm", "high", "watch", "low", "na"] as Level[]).map((l) => (
            <span key={l} className="count"><i style={{ background: LEVELS[l].color }} />{LEVELS[l].label} <b className="num">{count(l)}</b></span>
          ))}
        </div>
        <div className="seg" role="group" aria-label="Sắp xếp">
          <button aria-pressed={sort === "risk"} onClick={() => setSort("risk")}>Theo nguy cơ</button>
          <button aria-pressed={sort === "bed"} onClick={() => setSort("bed")}>Theo giường</button>
        </div>
      </div>
      {!rows.length ? <div className="card empty">Đang kết nối luồng dữ liệu…</div> : (
        <div className="tiles">
          {rows.map(({ b, s }) => (
            <BedTile key={b.bed} info={b} snap={s} buf={buffers.get(b.bed)} model={model} horizon={horizon} thr={thr} onOpen={onOpen} />
          ))}
        </div>
      )}
    </>
  );
}
