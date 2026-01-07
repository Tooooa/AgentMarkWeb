import React from 'react';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import type { Step } from '../../data/mockData';
import { Brain, Terminal, EyeOff, RotateCcw, ArrowRight } from 'lucide-react';
import { useI18n } from '../../i18n/I18nContext';
import { useState } from 'react';
import StepDetailModal from './StepDetailModal';

interface StepCardProps {
    step: Step;
    isErased: boolean;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-slate-800 text-white text-xs p-2 rounded shadow-lg border border-slate-700">
                <p className="font-bold mb-1">{payload[0].payload.name}</p>
                <p>Prob: {(payload[0].value * 100).toFixed(1)}%</p>
            </div>
        );
    }
    return null;
};

const StepCard: React.FC<StepCardProps> = ({ step, isErased }) => {
    const { t } = useI18n();
    const [isDetailOpen, setIsDetailOpen] = useState(false);

    if (isErased) {
        return (
            <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="relative overflow-hidden rounded-xl border border-rose-200 bg-rose-50/50 p-4 min-h-[120px] flex items-center justify-center"
            >
                {/* Glitch Effect Elements */}
                <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mixed-blend-overlay"></div>
                <div className="flex flex-col items-center gap-2 z-10">
                    <EyeOff size={32} className="text-rose-400 animate-pulse" />
                    <span className="font-mono font-bold text-rose-500 tracking-[0.2em] text-sm animate-pulse">{t('logDestroyed')}</span>
                    <span className="font-mono text-[10px] text-rose-400">ID: #{step.stepIndex} :: CHECKSUM_FAIL</span>
                </div>
            </motion.div>
        );
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel p-0 overflow-hidden group"
        >
            {/* Header */}
            <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between bg-white/40">
                <div className="flex items-center gap-2">
                    <span className="h-6 w-6 rounded-full bg-slate-200 text-slate-500 flex items-center justify-center text-xs font-bold font-mono">
                        {step.stepIndex}
                    </span>
                    <span className="text-xs font-semibold uppercase text-slate-400 tracking-wider">
                        {t('step')} #{step.stepIndex}
                    </span>
                </div>
                <div className={`text-xs px-2 py-1 rounded-full font-medium ${step.stepType === 'finish' ? 'bg-emerald-100 text-emerald-600' : 'bg-indigo-100 text-indigo-600'}`}>
                    {step.stepType.toUpperCase()}
                </div>
            </div>

            <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Left: Thought & Action */}
                <div className="space-y-3">
                    <div className="space-y-1">
                        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">
                            <Brain size={12} /> Thought
                        </div>
                        <p className="text-sm text-slate-600 italic leading-relaxed bg-slate-50/50 p-2 rounded-lg border border-slate-100/50">
                            {step.thought || "No thought process recorded."}
                        </p>
                    </div>

                    <div className="space-y-1">
                        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 uppercase tracking-wide">
                            <Terminal size={12} /> Action
                        </div>
                        <div className="font-mono text-sm text-slate-800 bg-slate-100 px-3 py-2 rounded-lg border border-slate-200 break-all">
                            {step.action}
                        </div>
                    </div>
                </div>

                {/* Right: Advanced Watermark Logic Visualization */}
                <div className="h-52 flex flex-col gap-2 bg-slate-50/50 rounded-lg p-2 border border-slate-100">
                    <div className="flex justify-between items-center text-[10px] text-slate-500 font-semibold uppercase tracking-wider px-1">
                        <span>Differential Decomposition</span>
                        <span>Recombination (Bins)</span>
                    </div>

                    <div className="flex-1 flex gap-2 items-center">
                        {/* Chart 1: Decomposition (Sorted Probabilities + Slicing) */}
                        <div className="flex-1 h-full relative">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={[...step.distribution].sort((a, b) => b.prob - a.prob)} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
                                    <XAxis dataKey="name" tick={false} />
                                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'transparent' }} />
                                    {/* Slicing Lines: Horizontal lines at each probability level */}
                                    {[...step.distribution].sort((a, b) => b.prob - a.prob).map((d, i) => (
                                        <ReferenceLine key={`line-${i}`} y={d.prob} stroke="#94a3b8" strokeDasharray="2 2" strokeOpacity={0.4} />
                                    ))}
                                    {/* Add minPointSize to ensure tiny prob bars are visible */}
                                    <Bar dataKey="prob" radius={[2, 2, 0, 0]} minPointSize={2}>
                                        {
                                            // Sort locally to ensure color mapping is correct if we wanted, but data is already sorted passed to chart
                                            [...step.distribution].sort((a, b) => b.prob - a.prob).map((entry, index) => (
                                                <Cell
                                                    key={`cell-d-${index}`}
                                                    fill={entry.isSelected ? '#818cf8' : '#e2e8f0'}
                                                    opacity={1}
                                                />
                                            ))
                                        }
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                            <div className="absolute bottom-1 right-2 text-[8px] text-slate-400 font-mono">Sorted Actions</div>
                        </div>

                        {/* Arrow Transition */}
                        <div className="text-slate-300">
                            <ArrowRight size={16} />
                        </div>

                        {/* Chart 2: Recombination (Bin Weights) */}
                        <div
                            className="flex-1 h-full relative group/chart cursor-pointer"
                            onClick={() => setIsDetailOpen(true)}
                        >
                            {(() => {
                                // 1. Calculate Bins dynamically
                                const sortedDist = [...step.distribution].sort((a, b) => b.prob - a.prob);
                                const bins = sortedDist.map((d, i) => {
                                    const nextProb = sortedDist[i + 1]?.prob || 0;
                                    const diff = d.prob - nextProb;
                                    return {
                                        name: `T_${i + 1}`,
                                        k: i + 1,
                                        weight: (i + 1) * diff,
                                        isTarget: d.isSelected // Simplification: we highlight the bin corresponding to the selected item's rank level
                                    };
                                }).filter(b => b.weight > 0.001); // Filter empty

                                return (
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={bins} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                                            <XAxis dataKey="name" tick={{ fontSize: 8 }} interval={0} />
                                            <Tooltip
                                                cursor={{ fill: 'transparent' }}
                                                content={({ active, payload }) => {
                                                    if (active && payload && payload.length) {
                                                        const data = payload[0].payload;
                                                        return (
                                                            <div className="bg-white/90 backdrop-blur p-2 shadow-lg rounded border border-slate-100 text-xs text-slate-700 z-50">
                                                                <p className="font-bold mb-1">{data.name}</p>
                                                                <p>Weight: {data.weight.toFixed(3)}</p>
                                                                <p>Size (K): {data.k}</p>
                                                            </div>
                                                        );
                                                    }
                                                    return null;
                                                }}
                                            />
                                            <Bar dataKey="weight" radius={[2, 2, 0, 0]}>
                                                {bins.map((entry, index) => (
                                                    <Cell
                                                        key={`cell-b-${index}`}
                                                        fill={entry.isTarget ? '#6366f1' : '#cbd5e1'} // Indigo-500 target, Slate-300 others
                                                        stroke={entry.isTarget ? '#312e81' : 'none'}
                                                        strokeWidth={entry.isTarget ? 2 : 0}
                                                    />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                );
                            })()}

                            {/* Cyclic Shift Indicator on top of the Recombination Chart */}
                            <motion.div
                                className="absolute top-0 right-0 bg-white/90 px-1.5 py-0.5 rounded shadow-sm border border-indigo-100 flex items-center gap-1"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: 0.5 }}
                            >
                                <RotateCcw size={10} className="text-indigo-600" />
                                <span className="text-[8px] font-mono text-indigo-800">Shift</span>
                            </motion.div>

                            {/* Hover Hint */}
                            <div className="absolute inset-0 bg-slate-900/0 hover:bg-slate-900/5 transition-colors flex items-center justify-center pointer-events-none opacity-0 group-hover/chart:opacity-100">
                                <span className="bg-white/90 px-2 py-1 rounded text-[10px] text-indigo-600 font-semibold shadow-sm backdrop-blur">
                                    Click to View Details
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Explanation Footer */}
                    <div className="bg-white border border-indigo-50 rounded p-1.5 flex items-center gap-2 text-[10px] text-slate-600 shadow-sm">
                        <div className="font-mono bg-indigo-100 text-indigo-700 px-1 rounded flex flex-col items-center leading-none py-0.5">
                            <span className="text-[6px] uppercase opacity-70">Payload</span>
                            <span className="font-bold">{step.watermark.bits}</span>
                        </div>

                        <div className="flex-1 flex items-center gap-1 overflow-hidden whitespace-nowrap">
                            <span className="text-slate-400">❶</span>Sort & Slice
                            <ArrowRight size={8} className="text-slate-300 mx-0.5" />
                            <span className="text-slate-400">❷</span>Select Bin <span className="font-bold text-indigo-600">T_{step.distribution.filter(x => x.prob >= (step.distribution.find(y => y.isSelected)?.prob || 0)).length}</span>
                            <ArrowRight size={8} className="text-slate-300 mx-0.5" />
                            <span className="text-slate-400">❸</span><RotateCcw size={8} className="inline mr-0.5" />Sample Action
                        </div>
                    </div>
                </div>
            </div>

            <StepDetailModal
                isOpen={isDetailOpen}
                onClose={() => setIsDetailOpen(false)}
                step={step}
            />
        </motion.div>
    );
};

export default StepCard;
