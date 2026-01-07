import React, { useEffect, useRef } from 'react';
import StepCard from './StepCard';
import type { Step } from '../../data/mockData';
import { useI18n } from '../../i18n/I18nContext';

interface FlowFeedProps {
    visibleSteps: Step[];
    erasedIndices: Set<number>;
}

const FlowFeed: React.FC<FlowFeedProps> = ({ visibleSteps, erasedIndices }) => {
    const bottomRef = useRef<HTMLDivElement>(null);
    const { t } = useI18n();

    // Auto-scroll to bottom when steps added
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [visibleSteps]);

    return (
        <div className="flex-1 overflow-y-auto px-1 py-2 space-y-4 h-full scrollbar-hide">
            {visibleSteps.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center text-slate-400 space-y-2">
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
