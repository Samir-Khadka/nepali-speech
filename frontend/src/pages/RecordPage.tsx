import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ApiError,
  fetchMine,
  fetchNextPrompt,
  getContributorId,
  pickMimeType,
  postConsent,
  submitRecording,
  type MineStats,
  type NextPrompt,
} from '../api';

const PROVINCES = [
  'कोशी प्रदेश',
  'मधेश प्रदेश',
  'बागमती प्रदेश',
  'गण्डकी प्रदेश',
  'लुम्बिनी प्रदेश',
  'कर्णाली प्रदेश',
  'सुदूरपश्चिम प्रदेश',
];

const AGE_RANGES = ['१८ भन्दा कम', '१८–२५', '२६–३५', '३६–५०', '५१ भन्दा माथि'];

const GENDERS = ['महिला', 'पुरुष', 'अन्य', 'भन्न चाहन्नँ'];

type Phase = 'idle' | 'recording' | 'recorded';

function ApiDown({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="error-box" role="alert">
      <strong>सर्भरमा जडान भएन / Couldn&apos;t reach the server.</strong>
      <p>
        कृपया इन्टरनेट जाँच गर्नुहोस् र केही बेरपछि फेरि प्रयास गर्नुहोस्। Please
        check your connection and try again.
      </p>
      <button type="button" className="btn btn-primary" onClick={onRetry}>
        फेरि प्रयास गर्नुहोस् · Retry
      </button>
    </div>
  );
}

export default function RecordPage() {
  const contributorId = useRef(getContributorId()).current;
  const [consented, setConsented] = useState<boolean>(
    () => localStorage.getItem('ns-consented') === '1',
  );
  const [province, setProvince] = useState('');
  const [ageRange, setAgeRange] = useState('');
  const [gender, setGender] = useState('');
  const [consentBusy, setConsentBusy] = useState(false);
  const [consentError, setConsentError] = useState<string | null>(null);

  const [prompt, setPrompt] = useState<NextPrompt | null>(null);
  const [mine, setMine] = useState<MineStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiDown, setApiDown] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [phase, setPhase] = useState<Phase>('idle');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const loadNext = useCallback(async () => {
    setLoading(true);
    setApiDown(false);
    setError(null);
    try {
      const [next, stats] = await Promise.all([
        fetchNextPrompt(contributorId),
        fetchMine(contributorId),
      ]);
      setPrompt(next);
      setMine(stats);
    } catch (err) {
      if (err instanceof TypeError) setApiDown(true);
      else if (err instanceof ApiError) setError(err.message);
      else setApiDown(true);
    } finally {
      setLoading(false);
    }
  }, [contributorId]);

  useEffect(() => {
    if (consented) void loadNext();
  }, [consented, loadNext]);

  // Stop any live mic stream when leaving the page.
  useEffect(
    () => () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    },
    [],
  );

  async function handleConsent(e: React.FormEvent) {
    e.preventDefault();
    setConsentBusy(true);
    setConsentError(null);
    try {
      await postConsent({
        contributor_id: contributorId,
        province: province || null,
        age_range: ageRange || null,
        gender: gender || null,
      });
      localStorage.setItem('ns-consented', '1');
      setConsented(true);
    } catch (err) {
      if (err instanceof TypeError) setConsentError('सर्भरमा जडान भएन। कृपया फेरि प्रयास गर्नुहोस्।');
      else if (err instanceof ApiError) setConsentError(err.message);
      else setConsentError('सहमति बुझाउन सकिएन। फेरि प्रयास गर्नुहोस्।');
    } finally {
      setConsentBusy(false);
    }
  }

  async function startRecording() {
    setError(null);
    setNotice(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('यो ब्राउजरले माइक्रोफोन रेकर्डिङ समर्थन गर्दैन। कृपया अर्को ब्राउजर प्रयास गर्नुहोस्।');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mimeType = pickMimeType();
      const rec = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      chunksRef.current = [];
      rec.ondataavailable = (ev: BlobEvent) => {
        if (ev.data.size > 0) chunksRef.current.push(ev.data);
      };
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, {
          type: rec.mimeType || 'audio/webm',
        });
        setAudioBlob(blob);
        setAudioUrl((old) => {
          if (old) URL.revokeObjectURL(old);
          return URL.createObjectURL(blob);
        });
        setPhase('recorded');
      };
      mediaRecorderRef.current = rec;
      rec.start();
      setPhase('recording');
    } catch {
      setError('माइक्रोफोन खोल्न सकिएन। कृपया माइक्रोफोन अनुमति दिनुहोस् र फेरि प्रयास गर्नुहोस्।');
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
  }

  function discardRecording() {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    setAudioBlob(null);
    setPhase('idle');
  }

  async function handleSubmit() {
    if (!prompt?.prompt_id || !audioBlob) return;
    setSubmitting(true);
    setError(null);
    try {
      await submitRecording({
        prompt_id: prompt.prompt_id,
        contributor_id: contributorId,
        audio: audioBlob,
      });
      discardRecording();
      setNotice('धन्यवाद! तपाईंको रेकर्डिङ बुझाइयो। Thank you — your recording was submitted.');
      await loadNext();
    } catch (err) {
      if (err instanceof TypeError) setApiDown(true);
      else if (err instanceof ApiError) {
        if (err.status === 403) {
          setError('सहमति आवश्यक छ। कृपया पृष्ठ ताजा गरेर सहमति फारम फेरि भर्नुहोस्।');
        } else {
          setError(err.message);
        }
      } else {
        setError('रेकर्डिङ बुझाउन सकिएन। फेरि प्रयास गर्नुहोस्।');
      }
    } finally {
      setSubmitting(false);
    }
  }

  if (!consented) {
    return (
      <div className="card">
        <h1>सहमति · Consent</h1>
        {/* Consent text — canonical wording from prompts/consent-ne.md (v1). */}
        <p>
          तपाईंले स्क्रिनमा देखाइएका नेपाली वाक्यहरू बोल्दै रेकर्ड गर्नुहुन्छ —
          हामी तपाईंको आवाज मात्र रेकर्ड गर्छौं। तपाईंका रेकर्डिङहरू एउटा खुला
          डाटासेटको हिस्सा बन्छन्, <strong>CC0</strong> अन्तर्गत सार्वजनिक
          गरिन्छ — जो कोहीले पनि कुनै पनि उद्देश्यका लागि, व्यावसायिक
          उद्देश्यसहित, बिना अनुमति र बिना श्रेय प्रयोग गर्न सक्छन्।
        </p>
        <p>
          तपाईंको रेकर्डिङसँग <strong>तपाईंको नाम, इमेल वा फोन नम्बर जोडिँदैन</strong> —
          ती विवरणहरू मागिँदैनन् र राखिँदैनन्। प्रदेश, उमेर समूह र लिङ्ग
          वैकल्पिक छन् र तपाईंको रेकर्डिङबाट छुट्टै राखिन्छन्। तपाईं{' '}
          <strong>जुनसुकै बेला रोक्न सक्नुहुन्छ</strong>, र आफ्ना रेकर्डिङहरू
          हटाउन अनुरोध गर्न सक्नुहुन्छ — हटाउने अनुरोध गर्ने माध्यम यो
          परियोजनाको GitHub रिपोजिटरीको Issues पृष्ठ हो। सहमति संस्करण: v1।
        </p>
        <p className="muted">
          By continuing, you agree that your recordings will be released under
          CC0 with no personal data attached, and that you may withdraw at any
          time.
        </p>
        <form onSubmit={handleConsent}>
          <label htmlFor="province">प्रदेश (वैकल्पिक · optional)</label>
          <select
            id="province"
            value={province}
            onChange={(e) => setProvince(e.target.value)}
          >
            <option value="">— छान्नुहोस् —</option>
            {PROVINCES.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>

          <label htmlFor="age">उमेर समूह (वैकल्पिक · optional)</label>
          <select id="age" value={ageRange} onChange={(e) => setAgeRange(e.target.value)}>
            <option value="">— छान्नुहोस् —</option>
            {AGE_RANGES.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>

          <label htmlFor="gender">लिङ्ग (वैकल्पिक · optional)</label>
          <select id="gender" value={gender} onChange={(e) => setGender(e.target.value)}>
            <option value="">— छान्नुहोस् —</option>
            {GENDERS.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>

          {consentError && (
            <div className="error-box" role="alert">
              {consentError}
            </div>
          )}
          <div className="btn-row">
            <button type="submit" className="btn btn-primary" disabled={consentBusy}>
              {consentBusy ? 'बुझाउँदै…' : 'म सहमत छु · I agree'}
            </button>
          </div>
        </form>
      </div>
    );
  }

  if (loading && !prompt) {
    return (
      <div className="card">
        <p>लोड हुँदैछ… Loading…</p>
      </div>
    );
  }

  if (apiDown) {
    return <ApiDown onRetry={() => void loadNext()} />;
  }

  if (prompt?.done || prompt?.prompt_id === null) {
    return (
      <div className="card">
        <h1>धन्यवाद! 🙏</h1>
        <p>
          तपाईंले सबै {prompt.total ?? ''} वाक्यहरू रेकर्ड गर्नुभयो। तपाईंको
          योगदानका लागि धेरै धेरै धन्यवाद!
        </p>
        <p className="muted">
          You have recorded every prompt. Thank you for contributing to the open
          Nepali speech dataset!
        </p>
        {mine && (
          <div className="progress">
            <div className="stat-chip">
              <strong>{mine.recorded}</strong>रेकर्ड गरियो
            </div>
            <div className="stat-chip">
              <strong>{mine.approved}</strong>स्वीकृत
            </div>
            <div className="stat-chip">
              <strong>{mine.pending}</strong>समीक्षामा
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <>
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

      {mine && (
        <div className="card">
          <h2>तपाईंको प्रगति · Your progress</h2>
          <div className="progress">
            <div className="stat-chip">
              <strong>{mine.recorded}</strong>रेकर्ड गरियो
            </div>
            <div className="stat-chip">
              <strong>{mine.approved}</strong>स्वीकृत
            </div>
            <div className="stat-chip">
              <strong>{mine.pending}</strong>समीक्षामा
            </div>
            <div className="stat-chip">
              <strong>{mine.rejected}</strong>अस्वीकृत
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <h1>वाक्य पढेर रेकर्ड गर्नुहोस्</h1>
        <p className="muted">
          तलको वाक्य स्पष्ट स्वरमा पढ्नुहोस्, त्यसपछि रेकर्ड थिच्नुहोस्। Read the
          sentence aloud, then tap Record.
        </p>
        {prompt && (
          <>
            <div>
              {prompt.domain && <span className="tag">{prompt.domain}</span>}
              {prompt.language && <span className="tag">{prompt.language}</span>}
            </div>
            <p className="prompt-text" lang="ne">
              {prompt.text}
            </p>
          </>
        )}

        {phase === 'idle' && (
          <button type="button" className="btn btn-primary btn-record" onClick={startRecording}>
            🎙️ रेकर्ड गर्नुहोस् · Record
          </button>
        )}
        {phase === 'recording' && (
          <button
            type="button"
            className="btn btn-record recording"
            onClick={stopRecording}
          >
            ⏹️ रोक्नुहोस् · Stop
          </button>
        )}
        {phase === 'recorded' && audioUrl && (
          <>
            <audio controls src={audioUrl} />
            <div className="btn-row">
              <button
                type="button"
                className="btn btn-success"
                onClick={handleSubmit}
                disabled={submitting}
              >
                {submitting ? 'बुझाउँदै…' : '✅ बुझाउनुहोस् · Submit'}
              </button>
              <button type="button" className="btn btn-outline" onClick={discardRecording}>
                🔄 फेरि रेकर्ड गर्नुहोस् · Re-record
              </button>
            </div>
          </>
        )}
      </div>

      {mine && mine.recordings.length > 0 && (
        <div className="card">
          <h2>हालका रेकर्डिङहरू</h2>
          <ul className="recording-list">
            {mine.recordings.slice(0, 5).map((r) => (
              <li key={r.recording_id}>
                <span className="status-badge">{r.status}</span>
                {r.prompt_text}
              </li>
            ))}
          </ul>
        </div>
      )}
    </>
  );
}
