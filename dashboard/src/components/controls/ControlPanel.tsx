import React from 'react';
import { Play, Pause, RotateCcw, SkipForward, SkipBack } from 'lucide-react';
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
    erasureRate,
    setErasureRate,
    currentStep,
    totalSteps,
    isLiveMode = false,
    onToggleLiveMode,
    apiKey = "",
    setApiKey,
    onInitSession
}) => {
    const { t, locale } = useI18n();
    const props = { isLiveMode, onToggleLiveMode, apiKey, setApiKey, onInitSession }; // Pack for cleaner usage below

    return (
        <div className="flex flex-col gap-4 h-full">
            {/* Scenario Selector */}
            <div className="glass p-5 rounded-2xl space-y-3">
                <label className="text-xs font-semibold uppercase text-slate-400 tracking-wider">{t('selectScenario')}</label>
                <select
                    className="w-full bg-slate-50 border border-slate-200 text-slate-700 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2.5 outline-none transition-all hover:bg-white"
                    value={activeScenarioId}
                    onChange={(e) => onSelectScenario(e.target.value)}
                >
                    {scenarios.map(s => (
                        <option key={s.id} value={s.id}>
                            {locale === 'zh' ? s.title.zh : s.title.en}
                        </option>
                    ))}
                </select>

                <div className="mt-2 text-xs text-slate-500">
                    {t('totalSteps')}: <span className="font-medium text-slate-700">{totalSteps}</span>
                </div>
            </div>

            {/* Playback Controls */}
            <div className="glass p-5 rounded-2xl space-y-4">
                <label className="text-xs font-semibold uppercase text-slate-400 tracking-wider">{t('playbackControl')}</label>

                <div className="flex items-center justify-center gap-4">
                    <button onClick={onPrev} className="p-2 hover:bg-slate-100 rounded-full text-slate-500 transition-colors" disabled={currentStep === 0}>
                        <SkipBack size={20} />
                    </button>

                    <button
                        onClick={onTogglePlay}
                        className={`p-4 rounded-full shadow-lg text-white transition-all transform hover:scale-105 active:scale-95 flex items-center justify-center ${isPlaying ? 'bg-amber-500 hover:bg-amber-600' : 'bg-indigo-600 hover:bg-indigo-700'}`}
                    >
                        {isPlaying ? <Pause size={24} fill="currentColor" /> : <Play size={24} fill="currentColor" className="ml-1" />}
                    </button>

                    <button onClick={onNext} className="p-2 hover:bg-slate-100 rounded-full text-slate-500 transition-colors" disabled={currentStep >= totalSteps}>
                        <SkipForward size={20} />
                    </button>
                </div>

                <div className="flex justify-center">
                    <button onClick={onReset} className="text-xs flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-100 text-slate-600 hover:bg-slate-200 transition-colors">
                        <RotateCcw size={12} /> {t('resetSimulation')}
                    </button>
                </div>

                {/* Progress Bar */}
                <div className="w-full bg-slate-200 rounded-full h-1.5 mt-2">
                    <div
                        className="bg-indigo-500 h-1.5 rounded-full transition-all duration-300 ease-out"
                        style={{ width: `${(currentStep / Math.max(totalSteps, 1)) * 100}%` }}
                    ></div>
                </div>
            </div>

            {/* Live Mode Toggle */}
            <div className="glass p-5 rounded-2xl space-y-3">
                <div className="flex items-center justify-between">
                    <label className="text-xs font-semibold uppercase text-indigo-500 tracking-wider">Live System</label>
                    <button
                        onClick={props.onToggleLiveMode}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${props.isLiveMode ? 'bg-indigo-600' : 'bg-slate-200'}`}
                    >
                        <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${props.isLiveMode ? 'translate-x-6' : 'translate-x-1'}`} />
                    </button>
                </div>

                {props.isLiveMode && (
                    <div className="space-y-2 animate-in fade-in slide-in-from-top-2">
                        <input
                            type="password"
                            placeholder="sk-..."
                            value={props.apiKey}
                            onChange={(e) => props.setApiKey?.(e.target.value)}
                            className="w-full bg-white border border-slate-200 text-xs rounded-lg px-3 py-2 outline-none focus:border-indigo-500"
                        />
                        <button
                            onClick={props.onInitSession}
                            disabled={!props.apiKey}
                            className="w-full bg-indigo-500 hover:bg-indigo-600 text-white text-xs font-medium py-2 rounded-lg transition-colors disabled:opacity-50"
                        >
                            Initialize Session
                        </button>
                    </div>
                )}
            </div>

            {/* Erasure Simulation */}
            <div className="glass p-5 rounded-2xl space-y-3">
                <div className="flex justify-between items-center">
                    <label className="text-xs font-semibold uppercase text-rose-500 tracking-wider">{t('logErasureRate')}</label>
                    <span className="text-sm font-bold text-rose-600">{erasureRate}%</span>
                </div>

                <input
                    type="range"
                    min="0"
                    max="50"
                    value={erasureRate}
                    onChange={(e) => setErasureRate(parseInt(e.target.value))}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-rose-500 hover:accent-rose-600"
                />

                <p className="text-xs text-slate-500 leading-relaxed">
                    {t('erasureDescription')}
                </p>
            </div>

        </div>
    );
};

export default ControlPanel;
