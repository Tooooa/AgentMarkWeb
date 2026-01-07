import React from 'react';
import { Play, Pause, RotateCcw, SkipForward, SkipBack, Activity, Zap, Columns } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { Trajectory } from '../../data/mockData';
import { useI18n } from '../../i18n/I18nContext';

interface ControlPanelProps {
    scenarios: Trajectory[];
    activeScenarioId: string;
    onSelectScenario: (id: string) => void;
    isPlaying: boolean;
    onTogglePlay: () => void;
    onReset: () => void;
    onNext: () => void;
    onPrev: () => void;
    erasureRate: number;
    setErasureRate: (val: number) => void;
    currentStep: number;
    totalSteps: number;
    isLiveMode?: boolean;
    onToggleLiveMode?: () => void;
    apiKey?: string;
    setApiKey?: (key: string) => void;
    onInitSession?: () => void;
    isComparisonMode?: boolean;
    onToggleComparisonMode?: () => void;
}

const ControlPanel: React.FC<ControlPanelProps> = ({
    scenarios,
    activeScenarioId,
    onSelectScenario,
    isPlaying,
    onTogglePlay,
    onReset,
    onNext,
    onPrev,
    currentStep,
    totalSteps,
    isComparisonMode,
    onToggleComparisonMode,
    isLiveMode,
    onToggleLiveMode
}) => {
    const { t, locale, setLocale } = useI18n();

    // Mock Data for Utility Monitor
    const tokenData = [
        { name: 'Base', value: 120 },
        { name: 'Ours', value: 125 },
    ];

    const latencyData = [
        { name: 'Base', value: 0.8 },
        { name: 'Ours', value: 0.85 },
    ];

    return (
        <div className="flex flex-col gap-6 h-full text-slate-800">
            {/* 1. Utility Monitor Section */}
            <div className="space-y-4">
                <div className="flex items-center gap-2 text-indigo-900 border-b border-indigo-100 pb-2">
                    <Activity size={18} />
                    <h3 className="font-bold text-sm uppercase tracking-wide">Utility Monitor</h3>
                </div>

                {/* Token Cost Chart */}
                <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-100">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-xs font-semibold text-slate-500">Token Cost</span>
                        <span className="text-[10px] bg-slate-100 px-1.5 py-0.5 rounded text-slate-500">tokens/sec</span>
                    </div>
                    <div className="h-32">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={tokenData} layout="vertical" margin={{ left: 0, right: 10, top: 0, bottom: 0 }}>
                                <XAxis type="number" hide />
                                <YAxis dataKey="name" type="category" width={30} tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                                <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ fontSize: '10px', borderRadius: '8px' }} />
                                <Bar dataKey="value" barSize={12} radius={[0, 4, 4, 0]}>
                                    {tokenData.map((_, index) => (
                                        <Cell key={`cell-${index}`} fill={index === 0 ? '#cbd5e1' : '#6366f1'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Latency Chart */}
                <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-100">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-xs font-semibold text-slate-500">Latency</span>
                        <span className="text-[10px] bg-slate-100 px-1.5 py-0.5 rounded text-slate-500">ms/step</span>
                    </div>
                    <div className="h-32">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={latencyData} layout="vertical" margin={{ left: 0, right: 10, top: 0, bottom: 0 }}>
                                <XAxis type="number" hide />
                                <YAxis dataKey="name" type="category" width={30} tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                                <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ fontSize: '10px', borderRadius: '8px' }} />
                                <Bar dataKey="value" barSize={12} radius={[0, 4, 4, 0]}>
                                    {latencyData.map((_, index) => (
                                        <Cell key={`cell-${index}`} fill={index === 0 ? '#cbd5e1' : '#f43f5e'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            <div className="flex-1"></div>

            {/* 2. Log Console / Playback Controls */}
            <div className="bg-slate-900 text-white p-5 rounded-2xl shadow-xl">
                <div className="flex justify-between items-center mb-4 border-b border-slate-700 pb-3">
                    <span className="font-mono text-xs text-slate-400">LOG CONSOLE</span>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setLocale(locale === 'zh' ? 'en' : 'zh')}
                            className="text-[10px] px-2 py-0.5 rounded border border-slate-600 text-slate-300 hover:text-white hover:border-slate-500 transition-colors"
                        >
                            {locale === 'zh' ? 'EN' : '中文'}
                        </button>
                        <Zap size={14} className="text-amber-400" />
                    </div>
                </div>

                {/* Live Mode Toggle */}
                <div className="flex gap-2 mb-2 bg-slate-800 p-1 rounded-lg">
                    <button
                        onClick={onToggleLiveMode}
                        className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-[10px] uppercase font-bold tracking-wider transition-all ${!isLiveMode ? 'bg-indigo-500 text-white shadow' : 'text-slate-400 hover:bg-slate-700 hover:text-slate-200'
                            }`}
                    >
                        Simulation
                    </button>
                    <button
                        onClick={onToggleLiveMode}
                        className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-[10px] uppercase font-bold tracking-wider transition-all ${isLiveMode ? 'bg-rose-500 text-white shadow' : 'text-slate-400 hover:bg-slate-700 hover:text-slate-200'
                            }`}
                    >
                        <Zap size={12} /> Live
                    </button>
                </div>

                {/* View Toggle */}
                <div className="flex gap-2 mb-4 bg-slate-800 p-1 rounded-lg">
                    <button
                        onClick={onToggleComparisonMode}
                        className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-[10px] uppercase font-bold tracking-wider transition-all ${!isComparisonMode ? 'bg-indigo-500 text-white shadow' : 'text-slate-400 hover:bg-slate-700 hover:text-slate-200'
                            }`}
                    >
                        <Activity size={12} /> Standard
                    </button>
                    <button
                        onClick={onToggleComparisonMode}
                        className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-[10px] uppercase font-bold tracking-wider transition-all ${isComparisonMode ? 'bg-indigo-500 text-white shadow' : 'text-slate-400 hover:bg-slate-700 hover:text-slate-200'
                            }`}
                    >
                        <Columns size={12} /> Compare
                    </button>
                </div>

                {/* Scenario Select (Compact) */}
                <div className="mb-4">
                    <select
                        className="w-full bg-slate-800 border border-slate-700 text-slate-300 text-xs rounded-lg p-2 outline-none focus:border-indigo-500"
                        value={activeScenarioId}
                        onChange={(e) => onSelectScenario(e.target.value)}
                    >
                        {scenarios.map(s => (
                            <option key={s.id} value={s.id}>
                                {locale === 'zh' ? s.title.zh : s.title.en}
                            </option>
                        ))}
                    </select>
                </div>

                {/* Playback Buttons */}
                <div className="flex items-center justify-between mb-4">
                    <button onClick={onPrev} className="p-2 hover:bg-slate-800 rounded-full text-slate-400 transition-colors" disabled={currentStep === 0}>
                        <SkipBack size={18} />
                    </button>

                    <button
                        onClick={onTogglePlay}
                        className={`p-3 rounded-full shadow-lg text-white transition-all active:scale-95 flex items-center justify-center ${isPlaying ? 'bg-amber-500 hover:bg-amber-600' : 'bg-indigo-600 hover:bg-indigo-700'}`}
                    >
                        {isPlaying ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" className="ml-0.5" />}
                    </button>

                    <button onClick={onNext} className="p-2 hover:bg-slate-800 rounded-full text-slate-400 transition-colors" disabled={currentStep >= totalSteps}>
                        <SkipForward size={18} />
                    </button>
                </div>

                {/* Progress */}
                <div className="space-y-1">
                    <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                        <span>STEP {currentStep}</span>
                        <span>{totalSteps} TOTAL</span>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1">
                        <div
                            className="bg-indigo-500 h-1 rounded-full transition-all duration-300"
                            style={{ width: `${(currentStep / Math.max(totalSteps, 1)) * 100}%` }}
                        ></div>
                    </div>
                </div>

                <div className="flex justify-center mt-4 pt-2 border-t border-slate-700/50">
                    <button onClick={onReset} className="text-[10px] flex items-center gap-1.5 px-3 py-1 rounded bg-slate-800 text-slate-400 hover:text-white transition-colors">
                        <RotateCcw size={10} /> {t('resetSimulation')}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ControlPanel;
