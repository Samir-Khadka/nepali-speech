import { useCallback, useEffect, useState } from 'react';
import {
  API_BASE,
  ApiError,
  fetchNextReview,
  getReviewerId,
  postReview,
  setReviewerId,
  type ReviewItem,
} from '../api';

export default function ReviewPage() {
  const [reviewerId, setReviewerIdState] = useState<string>(() => getReviewerId());
  const [savedReviewer, setSavedReviewer] = useState<string>(() => getReviewerId());
  const [item, setItem] = useState<ReviewItem | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiDown, setApiDown] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);

  const loadNext = useCallback(async (reviewer: string) => {
    if (!reviewer.trim()) return;
    setLoading(true);
    setApiDown(false);
    setError(null);
    setReason('');
    try {
      const next = await fetchNextReview(reviewer.trim());
      setItem(next);
    } catch (err) {
      if (err instanceof TypeError) setApiDown(true);
      else if (err instanceof ApiError) setError(err.message);
      else setApiDown(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (savedReviewer) void loadNext(savedReviewer);
  }, [savedReviewer, loadNext]);

  function saveReviewer(e: React.FormEvent) {
    e.preventDefault();
    const id = reviewerId.trim();
    if (!id) return;
    setReviewerId(id);
    setSavedReviewer(id);
  }

  async function decide(decision: 'approve' | 'reject') {
    if (!item?.recording_id || !savedReviewer) return;
    if (decision === 'reject' && !reason.trim()) {
      setError('अस्वीकार गर्दा कारण लेख्नु अनिवार्य छ। A reason is required when rejecting.');
      return;
    }
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const res = await postReview({
        recording_id: item.recording_id,
        reviewer_id: savedReviewer,
        decision,
        reason: decision === 'reject' ? reason.trim() : null,
      });
      setNotice(
        decision === 'approve'
          ? `स्वीकृत गरियो — नयाँ स्थिति: ${res.status}`
          : `अस्वीकृत गरियो — नयाँ स्थिति: ${res.status}`,
      );
      await loadNext(savedReviewer);
    } catch (err) {
      if (err instanceof TypeError) setApiDown(true);
      else if (err instanceof ApiError) setError(err.message);
      else setError('समीक्षा बुझाउन सकिएन। फेरि प्रयास गर्नुहोस्।');
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="card">
        <h1>समीक्षा · Review queue</h1>
        <p className="muted">
          रेकर्डिङ सुन्नुहोस् र वाक्यसँग मिल्छ कि छैन जाँच्नुहोस्। Listen to each
          recording and check it matches the prompt text.
        </p>
        <form onSubmit={saveReviewer}>
          <label htmlFor="reviewer-id">समीक्षक ID · Reviewer ID</label>
          <input
            id="reviewer-id"
            type="text"
            value={reviewerId}
            onChange={(e) => setReviewerIdState(e.target.value)}
            placeholder="e.g. reviewer-ram"
            autoComplete="off"
          />
          <div className="btn-row">
            <button type="submit" className="btn btn-primary">
              समीक्षा सुरु गर्नुहोस् · Start reviewing
            </button>
          </div>
        </form>
      </div>

      {apiDown && (
        <div className="error-box" role="alert">
          <strong>सर्भरमा जडान भएन / Couldn&apos;t reach the server.</strong>
          <p>कृपया फेरि प्रयास गर्नुहोस्। Please try again.</p>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => void loadNext(savedReviewer)}
          >
            फेरि प्रयास गर्नुहोस् · Retry
          </button>
        </div>
      )}

      {notice && (
        <div className="success-box" role="status">
          {notice}
        </div>
      )}
      {error && (
        <div className="error-box" role="alert">
          {error}
        </div>
      )}

      {savedReviewer && loading && (
        <div className="card">
          <p>लोड हुँदैछ… Loading…</p>
        </div>
      )}

      {savedReviewer && !loading && !apiDown && item?.empty && (
        <div className="card">
          <h2>सूची खाली छ 🎉</h2>
          <p>{item.message ?? 'समीक्षाका लागि अहिले कुनै रेकर्डिङ छैन।'}</p>
          <p className="muted">
            No recordings are waiting for review right now. Please check back
            later.
          </p>
          <div className="btn-row">
            <button
              type="button"
              className="btn btn-outline"
              onClick={() => void loadNext(savedReviewer)}
            >
              ताजा गर्नुहोस् · Refresh
            </button>
          </div>
        </div>
      )}

      {savedReviewer && !loading && !apiDown && item?.recording_id && (
        <div className="card">
          <div>
            {item.domain && <span className="tag">{item.domain}</span>}
            {item.language && <span className="tag">{item.language}</span>}
          </div>
          <p className="prompt-text" lang="ne">
            {item.prompt_text}
          </p>
          {item.audio_url && (
            <audio controls src={`${API_BASE}${item.audio_url}`} preload="auto" />
          )}
          <label htmlFor="reject-reason">
            अस्वीकारको कारण (अस्वीकार गर्दा अनिवार्य · required on reject)
          </label>
          <textarea
            id="reject-reason"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="e.g. गलत वाक्य पढियो / आवाज स्पष्ट छैन"
          />
          <div className="btn-row">
            <button
              type="button"
              className="btn btn-success"
              onClick={() => void decide('approve')}
              disabled={busy}
            >
              ✅ स्वीकृत · Approve
            </button>
            <button
              type="button"
              className="btn btn-danger"
              onClick={() => void decide('reject')}
              disabled={busy}
            >
              ❌ अस्वीकृत · Reject
            </button>
            <button
              type="button"
              className="btn btn-outline"
              onClick={() => void loadNext(savedReviewer)}
              disabled={busy}
            >
              ⏭️ छोड्नुहोस् · Skip
            </button>
          </div>
        </div>
      )}
    </>
  );
}
