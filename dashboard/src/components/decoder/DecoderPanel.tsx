import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Step } from '../../data/mockData';
import { CheckCircle2, Lock, ShieldCheck, WifiOff } from 'lucide-react';
import { useI18n } from '../../i18n/I18nContext';

interface DecoderPanelProps {
    visibleSteps: Step[];
    erasedIndices: Set<number>;
    targetPayload?: string;
}

// Fixed payload length for demo
const REQUIRED_RANK = 16;

const DecoderPanel: React.FC<DecoderPanelProps> = ({ visibleSteps, erasedIndices, targetPayload }) => {
    const { t } = useI18n();

    // Filter only valid (non-erased) steps that contribute to the matrix
    const validSteps = visibleSteps.filter(s => !erasedIndices.has(s.stepIndex));
    // Count total equations produced by valid steps
    const currentRank = Math.min(
        validSteps.reduce((acc, step) => acc + (step.watermark.matrixRows?.length || 0), 0),
        REQUIRED_RANK
    );
    const progress = (currentRank / REQUIRED_RANK) * 100;
    const isSuccess = currentRank >= REQUIRED_RANK;

    return (
        <div className="flex flex-col gap-4 h-full">

            {/* Status Card */}
            <div className={`glass p-6 rounded-2xl relative overflow-hidden transition-all duration-500 ${isSuccess ? 'border-emerald-200 bg-emerald-50/50' : ''}`}>
                {/* Background Pattern */}
                <div className="absolute top-0 right-0 p-4 opacity-5">
                    <ShieldCheck size={120} />
                </div>

                <div className="flex items-center justify-between mb-4">
                    <div>
                        <h2 className="text-lg font-bold text-slate-800">{t('decoderTitle')}</h2>
                        <p className="text-xs text-slate-500">{t('decoderSubtitle')}</p>
                    </div>
                    <div className={`h-10 w-10 rounded-full flex items-center justify-center ${isSuccess ? 'bg-emerald-500 text-white' : 'bg-slate-200 text-slate-400'}`}>
                        {isSuccess ? <CheckCircle2 size={20} /> : <Lock size={18} />}
                    </div>
                </div>

                <div className="space-y-2">
                    <div className="flex justify-between text-xs font-semibold uppercase tracking-wider text-slate-600">
                        <span>{t('decodingProgress')}</span>
                        <span>{currentRank} / {REQUIRED_RANK} ({t('rank')})</span>
                    </div>
                    <div className="h-3 w-full bg-slate-200 rounded-full overflow-hidden">
                        <div
                            className={`h-full transition-all duration-700 ease-out ${isSuccess ? 'bg-emerald-500' : 'bg-indigo-500'}`}
                            style={{ width: `${progress}%` }}
                        ></div>
                    </div>
                </div>

                {isSuccess && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mt-4 p-3 bg-emerald-100/50 rounded-lg border border-emerald-200 text-emerald-800 text-sm font-mono flex items-center justify-center gap-2"
                    >
                        <span>{t('payloadDecoded')}:</span>
                        <span className="font-bold tracking-[0.2em] bg-white px-2 py-0.5 rounded shadow-sm">{targetPayload || "11001101"}</span>
                    </motion.div>
                )}
            </div>

            {/* Matrix Visualization */}
            <div className="glass flex-1 rounded-2xl p-0 flex flex-col overflow-hidden">
                <div className="p-4 border-b border-slate-100/50 bg-slate-50/50 backdrop-blur-md z-10">
                    <h3 className="text-xs font-bold uppercase text-slate-500 tracking-wider">{t('matrixTitle')}</h3>
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-1 scrollbar-hide bg-slate-50/30">
                    <AnimatePresence>
                        {visibleSteps.map((step) => {
                            const isErased = erasedIndices.has(step.stepIndex);
                            return (
                                <motion.div
                                    key={step.stepIndex}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0 }}
                                    className={`w-full rounded mb-2 border transition-all overflow-hidden ${isErased
                                        ? 'bg-rose-50 border-rose-100 text-rose-300'
                                        : 'bg-white border-white/50 shadow-sm'
                                        }`}
                                >
                                    <div className="flex items-start">
                                        {/* Step Header / ID */}
                                        <div className="w-10 p-2 text-[10px] font-mono text-slate-400 border-r border-slate-100/50 flex flex-col items-center justify-center bg-slate-50/50">
                                            <span>#{step.stepIndex}</span>
                                            <span className="text-[8px] opacity-70">S{step.stepIndex}</span>
                                        </div>

                                        {/* Step Content (Equations) */}
                                        <div className="flex-1 p-2 flex flex-col gap-1.5">
                                            {isErased ? (
                                                <div className="flex justify-center items-center py-1 gap-2 text-[10px] font-medium tracking-widest text-rose-400">
                                                    <WifiOff size={12} /> {t('packetLost')}
                                                </div>
                                            ) : (
                                                // Map through EACH equation/row generated by this step
                                                step.watermark.matrixRows.map((row, rowIndex) => (
                                                    <div key={`${step.stepIndex}-${rowIndex}`} className="flex justify-between items-center px-1">
                                                        {/* Row Index (optional, kept subtle) */}
                                                        {/* <span className="text-[8px] text-slate-300 mr-2 w-3">{rowIndex + 1}</span> */}

                                                        {/* The Dots */}
                                                        <div className="flex-1 flex justify-between">
                                                            {row.map((bit, colIndex) => (
                                                                <div
                                                                    key={colIndex}
                                                                    className={`w-1.5 h-1.5 rounded-full transition-colors duration-300 ${bit ? 'bg-indigo-500' : 'bg-slate-200'}`}
                                                                    title={`Bit ${colIndex}`}
                                                                ></div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                ))
                                            )}
                                        </div>
                                    </div>
                                </motion.div>
                            );
                        })}
                    </AnimatePresence>

                    {visibleSteps.length === 0 && (
                        <div className="h-full flex items-center justify-center text-slate-300 text-xs italic">
                            {t('matrixEmpty')}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default DecoderPanel;
