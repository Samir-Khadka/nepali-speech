// API client for the nepali-speech Phase 1 contract (see ../CONTRACT.md).
// Base URL from VITE_API_URL, default http://localhost:8000.

export const API_BASE: string =
  (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';

export function getContributorId(): string {
  let id = localStorage.getItem('ns-contributor-id');
  if (!id) {
    id =
      typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : `ns-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    localStorage.setItem('ns-contributor-id', id);
  }
  return id;
}

export function getReviewerId(): string {
  return localStorage.getItem('ns-reviewer-id') ?? '';
}

export function setReviewerId(id: string): void {
  localStorage.setItem('ns-reviewer-id', id);
}

export interface NextPrompt {
  prompt_id: number | null;
  text?: string;
  domain?: string;
  language?: string;
  done?: boolean;
  total?: number;
}

export interface MineStats {
  recorded: number;
  approved: number;
  pending: number;
  rejected: number;
  recordings: { recording_id: number; prompt_text: string; status: string }[];
}

export interface ReviewItem {
  recording_id?: number;
  prompt_text?: string;
  domain?: string;
  language?: string;
  audio_url?: string;
  empty?: boolean;
  message?: string;
}

export interface AdminStats {
  prompts: number;
  contributors: number;
  recordings_total: number;
  by_status: Record<string, number>;
  reviews_total: number;
  double_reviewed: number;
  agreement_rate: number | null;
  by_domain: Record<string, number>;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function handleResponse(res: Response): Promise<unknown> {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = (await res.json()) as { detail?: string; message?: string };
      if (typeof body.detail === 'string') detail = body.detail;
      else if (typeof body.message === 'string') detail = body.message;
    } catch {
      // keep generic detail
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<unknown>;
}

export async function fetchNextPrompt(contributorId: string): Promise<NextPrompt> {
  const res = await fetch(
    `${API_BASE}/api/prompts/next?contributor_id=${encodeURIComponent(contributorId)}`,
  );
  return (await handleResponse(res)) as NextPrompt;
}

export async function postConsent(input: {
  contributor_id: string;
  province: string | null;
  age_range: string | null;
  gender: string | null;
}): Promise<{ ok: boolean; contributor_id: string }> {
  const res = await fetch(`${API_BASE}/api/consent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...input, version: 'v1' }),
  });
  return (await handleResponse(res)) as { ok: boolean; contributor_id: string };
}

export async function submitRecording(input: {
  prompt_id: number;
  contributor_id: string;
  audio: Blob;
}): Promise<{ recording_id: number; status: string }> {
  const form = new FormData();
  form.append('prompt_id', String(input.prompt_id));
  form.append('contributor_id', input.contributor_id);
  const ext =
    input.audio.type.includes('mp4') || input.audio.type.includes('mpeg')
      ? 'mp4'
      : input.audio.type.includes('ogg')
        ? 'ogg'
        : input.audio.type.includes('wav')
          ? 'wav'
          : 'webm';
  form.append('audio', input.audio, `recording.${ext}`);
  const res = await fetch(`${API_BASE}/api/recordings`, {
    method: 'POST',
    body: form,
  });
  return (await handleResponse(res)) as { recording_id: number; status: string };
}

export async function fetchMine(contributorId: string): Promise<MineStats> {
  const res = await fetch(
    `${API_BASE}/api/recordings/mine?contributor_id=${encodeURIComponent(contributorId)}`,
  );
  return (await handleResponse(res)) as MineStats;
}

export async function fetchNextReview(reviewerId: string): Promise<ReviewItem> {
  const res = await fetch(
    `${API_BASE}/api/review/next?reviewer_id=${encodeURIComponent(reviewerId)}`,
  );
  return (await handleResponse(res)) as ReviewItem;
}

export async function postReview(input: {
  recording_id: number;
  reviewer_id: string;
  decision: 'approve' | 'reject';
  reason: string | null;
}): Promise<{ ok: boolean; status: string }> {
  const res = await fetch(`${API_BASE}/api/review/${input.recording_id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      reviewer_id: input.reviewer_id,
      decision: input.decision,
      reason: input.reason,
    }),
  });
  return (await handleResponse(res)) as { ok: boolean; status: string };
}

export async function fetchAdminStats(adminEmail: string): Promise<AdminStats> {
  const res = await fetch(`${API_BASE}/api/admin/stats`, {
    headers: { 'X-Admin-Email': adminEmail },
  });
  return (await handleResponse(res)) as AdminStats;
}

/** Pick the first MediaRecorder mime type the browser supports. */
export function pickMimeType(): string {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
    'audio/ogg',
  ];
  if (typeof MediaRecorder === 'undefined') return '';
  for (const mime of candidates) {
    try {
      if (MediaRecorder.isTypeSupported(mime)) return mime;
    } catch {
      // try next
    }
  }
  return '';
}
