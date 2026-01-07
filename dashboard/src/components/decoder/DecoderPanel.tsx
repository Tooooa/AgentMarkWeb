import React, { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Step } from '../../data/mockData';
import { CheckCircle2, Lock, Database, FileDigit } from 'lucide-react';
import { useI18n } from '../../i18n/I18nContext';

interface DecoderPanelProps {
    visibleSteps: Step[];
    erasedIndices: Set<number>;
    targetPayload?: string;
}

const REQUIRED_RANK = 16;

const DecoderPanel: React.FC<DecoderPanelProps> = ({ visibleSteps, erasedIndices, targetPayload }) => {
    const { locale } = useI18n();
    const bottomRef = useRef<HTMLDivElement>(null);

    // Auto-scroll
    useEffect(() => {
        if (visibleSteps.length > 0) {
            bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
        }
    }, [visibleSteps]);

    const validSteps = visibleSteps.filter(s => !erasedIndices.has(s.stepIndex));
    const currentRank = Math.min(
        validSteps.reduce((acc, step) => acc + (step.watermark.matrixRows?.length || 0), 0),
        REQUIRED_RANK
    );
    const progress = (currentRank / REQUIRED_RANK) * 100;
    const isSuccess = currentRank >= REQUIRED_RANK;

    return (
        <div className="flex flex-col gap-4 h-full bg-slate-50/50">

            {/* 1. Decoding Status / Progress */}
            <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100 flex flex-col gap-3">
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="font-bold text-slate-800 text-sm">{locale === 'zh' ? '解码进度' : 'Decoding Progress'}</h2>
                        <span className="text-xs text-slate-400 font-mono">RANK: {currentRank}/{REQUIRED_RANK}</span>
                    </div>
                    {isSuccess ? (
                        <div className="bg-emerald-100 text-emerald-600 px-2 py-1 rounded-lg flex items-center gap-1 text-xs font-bold">
                            <CheckCircle2 size={14} /> <span>SUCCESS</span>
                        </div>
                    ) : (
                        <div className="bg-indigo-50 text-indigo-500 px-2 py-1 rounded-lg flex items-center gap-1 text-xs font-bold">
                            <Lock size={14} /> <span>LOCKED</span>
                        </div>
                    )}
                </div>

                {/* Progress Bar */}
                <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                    <motion.div
                        className={`h-full ${isSuccess ? 'bg-emerald-500' : 'bg-indigo-500'}`}
                        initial={{ width: 0 }}
                        animate={{ width: `${progress}%` }}
                        transition={{ duration: 0.5 }}
                    />
                </div>

                {isSuccess && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        className="bg-emerald-50 rounded border border-emerald-100 p-2 text-center"
                    >
                        <p className="text-[10px] text-emerald-600 uppercase tracking-wider mb-1">Payload Recovered</p>
                        <p className="font-mono text-lg font-bold text-emerald-800 tracking-widest">{targetPayload || "AGENTMARK"}</p>
                    </motion.div>
                )}
            </div>

            {/* 2. Received Datasets List */}
            <div className="flex-1 overflow-hidden flex flex-col bg-white rounded-2xl shadow-sm border border-slate-100">
                <div className="p-4 border-b border-slate-50 bg-slate-50/30 flex items-center gap-2">
                    <Database size={14} className="text-slate-400" />
                    <h3 className="font-bold text-xs text-slate-500 uppercase tracking-wider">
                        {locale === 'zh' ? '接收到的数据集' : 'Received Datasets'}
                    </h3>
                    <span className="ml-auto bg-slate-200 text-slate-600 text-[10px] px-1.5 py-0.5 rounded-full font-mono">
                        {visibleSteps.length}
                    </span>
                </div>

                <div className="flex-1 overflow-y-auto p-3 space-y-2 scrollbar-hide">
                    <AnimatePresence>
                        {visibleSteps.map((step) => {
                            const isErased = erasedIndices.has(step.stepIndex);
                            return (
                                <motion.div
                                    key={step.stepIndex}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    className={`relative p-3 rounded-xl border transition-all group ${isErased
                                        ? 'bg-rose-50 border-rose-100 opacity-70 grayscale-[0.5]'
                                        : 'bg-white border-slate-100 hover:border-indigo-200 hover:shadow-sm'}`}
                                >
                                    <div className="flex items-start gap-3">
                                        {/* Icon Box */}
                                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${isErased ? 'bg-rose-100 text-rose-400' : 'bg-indigo-50 text-indigo-500'}`}>
                                            <FileDigit size={16} />
                                        </div>

                                        <div className="flex-1 min-w-0">
                                            <div className="flex justify-between items-center mb-1">
                                                <span className={`text-xs font-bold ${isErased ? 'text-rose-400' : 'text-slate-700'}`}>
                                                    Dataset #{step.stepIndex}
                                                </span>
                                                <span className="text-[10px] font-mono text-slate-400">
                                                    {"14:02:23"}
                                                </span>
                                            </div>

                                            {isErased ? (
                                                <div className="text-[10px] text-rose-400 font-medium bg-rose-100/50 px-2 py-1 rounded inline-block">
                                                    PACKET LOST
                                                </div>
                                            ) : (
                                                <div className="space-y-1.5">
                                                    {/* Matrix Bits Visual */}
                                                    {step.watermark.matrixRows.map((row, rIdx) => (
                                                        <div key={rIdx} className="flex items-center gap-1">
                                                            <span className="text-[8px] text-slate-300 w-3">R{rIdx}</span>
                                                            <div className="flex gap-1">
                                                                {row.map((bit, cIdx) => (
                                                                    <div
                                                                        key={cIdx}
                                                                        className={`w-1.5 h-1.5 rounded-full ${bit ? 'bg-indigo-500' : 'bg-slate-200'}`}
                                                                    />
                                                                ))}
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </motion.div>
                            );
                        })}
                    </AnimatePresence>
                    <div ref={bottomRef} className="h-1" />
                </div>
            </div>
        </div>
    );
};

export default DecoderPanel;
