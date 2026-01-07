import React, { useEffect, useRef } from 'react';
import StepCard from './StepCard';
import type { Step } from '../../data/mockData';
import { useI18n } from '../../i18n/I18nContext';
import { User, Bot } from 'lucide-react';

interface FlowFeedProps {
    visibleSteps: Step[];
    erasedIndices: Set<number>;
    userQuery: string;
}

const FlowFeed: React.FC<FlowFeedProps> = ({ visibleSteps, erasedIndices, userQuery }) => {
    const bottomRef = useRef<HTMLDivElement>(null);
    const { t } = useI18n();

    // Auto-scroll to bottom when steps added
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [visibleSteps]);

    return (
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-8 h-full scrollbar-hide">
            {/* User Query Bubble */}
            <div className="flex gap-4">
                <div className="flex-shrink-0 mt-1">
                    <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center text-slate-500">
                        <User size={18} />
                    </div>
                </div>
                <div className="flex-1">
                    <div className="bg-slate-100 rounded-2xl rounded-tl-none p-4 text-slate-800 text-sm shadow-sm inline-block max-w-[90%]">
                        <p className="font-semibold text-xs text-slate-500 mb-1 uppercase tracking-wide">User Prompt</p>
                        {userQuery || "No query provided."}
                    </div>
                </div>
            </div>

            {visibleSteps.length === 0 && (
                <div className="flex flex-col items-center justify-center text-slate-400 space-y-2 py-10">
                    <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center animate-pulse">
                        <Bot size={20} className="text-slate-300" />
                    </div>
                    <p className="text-sm">{t('waiting')}</p>
                </div>
            )}

            {visibleSteps.map((step) => (
                <StepCard
                    key={step.stepIndex}
                    step={step}
                    isErased={erasedIndices.has(step.stepIndex)}
                />
            ))}
            <div ref={bottomRef} className="h-4" />
        </div>
    );
};

export default FlowFeed;
