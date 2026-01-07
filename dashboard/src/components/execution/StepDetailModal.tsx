
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine, CartesianGrid } from 'recharts';
import { X, RotateCcw, ArrowRight } from 'lucide-react';
import type { Step } from '../../data/mockData';

interface StepDetailModalProps {
    isOpen: boolean;
    onClose: () => void;
    step: Step;
}

const StepDetailModal: React.FC<StepDetailModalProps> = ({ isOpen, onClose, step }) => {

    if (!isOpen) return null;

    // --- Data Preparation ---
    const sortedDist = [...step.distribution].sort((a, b) => b.prob - a.prob);

    // 1. Decomposition Data (Left Chart)
    const decompositionData = sortedDist;

    // 2. Recombination Data (Right Chart - Stacked)
    const binsData = sortedDist.map((d, k) => { // k is 0-indexed here (Rank k+1)
        const nextProb = sortedDist[k + 1]?.prob || 0;
        const diff = d.prob - nextProb;

        if (diff <= 0.000001) return null; // Skip empty bins

        const binObj: any = {
            name: `T_${k + 1}`,
            totalWeight: (k + 1) * diff,
            k: k + 1,
            isTarget: d.isSelected,
        };

        // Add stacking components
        // For j=0 to k: component "Action_{j+1}" = diff
        for (let j = 0; j <= k; j++) {
            binObj[`Action_${j + 1}`] = diff;
        }

        return binObj;
    }).filter(b => b !== null);

    // Colors for Ranks (Stack layers)
    const getRankColor = (rankIndex: number) => {
        const colors = [
            '#f87171', // Red
            '#fbbf24', // Amber
            '#34d399', // Emerald
            '#60a5fa', // Blue
            '#818cf8', // Indigo
            '#a78bfa', // Violet
            '#f472b6', // Pink
        ];
        return colors[rankIndex % colors.length];
    };

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-sm p-4"
                onClick={onClose}
            >
                <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.9, opacity: 0 }}
                    className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl max-h-[90vh] overflow-hidden flex flex-col"
                    onClick={e => e.stopPropagation()}
                >
                    {/* Header */}
                    <div className="flex justify-between items-center p-6 border-b border-slate-100">
                        <div>
                            <h2 className="text-xl font-bold text-slate-800">Differential Sampling Visualization</h2>
                            <p className="text-sm text-slate-500">Step #{step.stepIndex}: {step.thought.substring(0, 80)}...</p>
                        </div>
                        <button onClick={onClose} className="p-2 hover:bg-slate-100 rounded-full transition-colors">
                            <X size={24} className="text-slate-500" />
                        </button>
                    </div>

                    {/* Content */}
                    <div className="flex-1 overflow-y-auto p-6 bg-slate-50/50">
                        <div className="flex gap-4 h-[500px]">
                            {/* Left Chart: Decomposition */}
                            <div className="flex-1 bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col">
                                <h3 className="text-sm font-bold text-slate-600 mb-4 uppercase tracking-wider text-center">
                                    1. Probability Decomposition (Sort & Slice)
                                </h3>
                                <div className="flex-1 relative">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={decompositionData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                            <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} angle={-45} textAnchor="end" height={60} />
                                            <YAxis />
                                            <Tooltip />
                                            {/* Slicing Lines */}
                                            {decompositionData.map((d, i) => (
                                                <ReferenceLine key={`line-${i}`} y={d.prob} stroke="#94a3b8" strokeDasharray="4 4" label={{ position: 'right', value: `P${i + 1}`, fontSize: 10, fill: '#94a3b8' }} />
                                            ))}
                                            <Bar dataKey="prob" radius={[4, 4, 0, 0]}>
                                                {decompositionData.map((_, i) => (
                                                    <Cell key={i} fill={getRankColor(i)} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* Arrow */}
                            <div className="flex items-center justify-center text-slate-300">
                                <ArrowRight size={48} />
                            </div>

                            {/* Right Chart: Recombination */}
                            <div className="flex-1 bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col">
                                <h3 className="text-sm font-bold text-slate-600 mb-4 uppercase tracking-wider text-center">
                                    2. Recombination (Stacked Bins)
                                </h3>
                                <div className="flex-1 relative">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={binsData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                            <XAxis dataKey="name" />
                                            <YAxis />
                                            <Tooltip cursor={{ fill: 'transparent' }} content={({ active, payload, label }) => {
                                                if (active && payload && payload.length) {
                                                    const bin = payload[0].payload;
                                                    return (
                                                        <div className="bg-white/95 backdrop-blur shadow-xl border border-slate-200 p-3 rounded-lg text-xs">
                                                            <p className="font-bold mb-2 text-base text-slate-800">{label}</p>
                                                            <div className="space-y-1">
                                                                <p className="flex justify-between gap-4"><span>Total Weight:</span> <span className="font-mono font-bold">{bin.totalWeight.toFixed(4)}</span></p>
                                                                <p className="flex justify-between gap-4"><span>Bin Size (K):</span> <span className="font-mono">{bin.k}</span></p>
                                                                <div className="border-t border-slate-100 my-1 pt-1">
                                                                    <p className="font-semibold text-slate-500 mb-1">Contains parts from:</p>
                                                                    {sortedDist.slice(0, bin.k).map((d, idx) => (
                                                                        <div key={idx} className="flex items-center gap-2">
                                                                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: getRankColor(idx) }}></div>
                                                                            <span className="truncate max-w-[150px]">{d.name}</span>
                                                                        </div>
                                                                    ))}
                                                                </div>
                                                            </div>
                                                        </div>
                                                    );
                                                }
                                                return null;
                                            }} />

                                            {/* Render Stacked Bars */}
                                            {/* We stack up to max K. Max K is sortedDist.length */}
                                            {sortedDist.map((_, i) => (
                                                <Bar
                                                    key={`stack-${i}`}
                                                    dataKey={`Action_${i + 1}`}
                                                    stackId="a"
                                                    fill={getRankColor(i)}
                                                    stroke="white"
                                                    strokeWidth={1}
                                                />
                                            ))}

                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>

                        {/* Explanation Text */}
                        <div className="mt-6 p-4 bg-indigo-50/50 rounded-xl border border-indigo-100 text-sm text-slate-600 space-y-2">
                            <h4 className="font-bold text-indigo-800 flex items-center gap-2">
                                <RotateCcw size={16} /> How Differential Sampling Works:
                            </h4>
                            <p>1. <strong>Sort & Slice:</strong> We sort all candidate actions by probability. Horizontal "slicing lines" are drawn at each probability level.</p>
                            <p>2. <strong>Create Bins:</strong> Each slice difference forms a "layer". Since k candidates have probability greater than or equal to this level, this layer is repeated k times to form Bin T_k.</p>
                            <p>3. <strong>Stacking:</strong> As shown in the right chart, Bin T_k is essentially a stack of small pieces from the top k actions. If the watermark selects Bin T_k, we then do a cyclic shift to pick one of these k actions.</p>
                        </div>
                    </div>
                </motion.div>
            </motion.div>
        </AnimatePresence>
    );
};

export default StepDetailModal;
