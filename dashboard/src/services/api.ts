
import axios from 'axios';

const API_BASE = '/api';

export interface Scenario {
    id: string;
    query: string;
}

export interface StepResponse {
    thought: string;
    action: string;
    observation: string;
    done: boolean;
    watermark?: {
        bits: string;
        matrixRows: number[][];
        rankContribution: number;
    };
    distribution?: {
        name: string;
        prob: number;
        isSelected: boolean;
    }[];
    stepIndex: number;
}

export const api = {
    initSession: async (apiKey: string, scenarioId: string) => {
        const response = await axios.post(`${API_BASE}/init`, { apiKey, scenarioId });
        return response.data; // { sessionId, task, totalSteps }
    },

    step: async (sessionId: string): Promise<StepResponse> => {
        const response = await axios.post(`${API_BASE}/step`, { sessionId });
        return response.data;
    }
};
