import { useState } from 'react';
import { ApiError, fetchAdminStats, type AdminStats } from '../api';

export default function StatsPage() {
  const [email, setEmail] = useState<string>(
    () => localStorage.getItem('ns-admin-email') ?? '',
  );
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiDown, setApiDown] = useState(false);
  const [notAdmin, setNotAdmin] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load(e: React.FormEvent) {
    e.preventDefault();
    const adminEmail = email.trim();
    if (!adminEmail) return;
    localStorage.setItem('ns-admin-email', adminEmail);
    setLoading(true);
    setApiDown(false);
    setNotAdmin(false);
    setError(null);
    try {
      const data = await fetchAdminStats(adminEmail);
      setStats(data);
    } catch (err) {
      if (err instanceof TypeError) setApiDown(true);
      else if (err instanceof ApiError && err.status === 403) setNotAdmin(true);
      else if (err instanceof ApiError) setError(err.message);
      else setApiDown(true);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <div className="card">
        <h1>तथ्याङ्क · Admin stats</h1>
        <p className="muted">
          यो पृष्ठ प्रशासकहरूका लागि मात्र हो। This page is for admins only.
        </p>
        <form onSubmit={load}>
          <label htmlFor="admin-email">प्रशासक इमेल · Admin email</label>
          <input
            id="admin-email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="admin@example.org"
            autoComplete="email"
          />
          <div className="btn-row">
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'लोड हुँदैछ…' : 'तथ्याङ्क हेर्नुहोस् · View stats'}
            </button>
          </div>
        </form>
      </div>

      {apiDown && (
        <div className="error-box" role="alert">
          <strong>सर्भरमा जडान भएन / Couldn&apos;t reach the server.</strong>
          <p>कृपया फेरि प्रयास गर्नुहोस्। Please try again.</p>
        </div>
      )}

      {notAdmin && (
        <div className="error-box" role="alert">
          <strong>तपाईं प्रशासक हुनुहुन्न।</strong>
          <p>
            यो इमेल प्रशासक सूचीमा छैन, त्यसैले तथ्याङ्क हेर्न मिल्दैन। You are
            not an admin — this email is not on the admin list.
          </p>
        </div>
      )}

      {error && (
        <div className="error-box" role="alert">
          {error}
        </div>
      )}

      {stats && (
        <>
          <div className="card">
            <h2>समग्र · Totals</h2>
            <div className="stat-grid">
              <div className="stat-chip">
                <strong>{stats.prompts}</strong>वाक्यहरू · Prompts
              </div>
              <div className="stat-chip">
                <strong>{stats.contributors}</strong>योगदानकर्ता · Contributors
              </div>
              <div className="stat-chip">
                <strong>{stats.recordings_total}</strong>रेकर्डिङहरू · Recordings
              </div>
              <div className="stat-chip">
                <strong>{stats.reviews_total}</strong>समीक्षाहरू · Reviews
              </div>
              <div className="stat-chip">
                <strong>{stats.double_reviewed}</strong>दोहोरो समीक्षा · Double-reviewed
              </div>
              <div className="stat-chip">
                <strong>
                  {stats.agreement_rate === null
                    ? '—'
                    : `${Math.round(stats.agreement_rate * 100)}%`}
                </strong>
                सहमति दर · Agreement rate
              </div>
            </div>
          </div>

          <div className="card">
            <h2>स्थिति अनुसार · By status</h2>
            <table className="stats-table">
              <tbody>
                {Object.entries(stats.by_status).map(([status, count]) => (
                  <tr key={status}>
                    <td>{status}</td>
                    <td>{count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card">
            <h2>विषय अनुसार · By domain</h2>
            <table className="stats-table">
              <tbody>
                {Object.entries(stats.by_domain).map(([domain, count]) => (
                  <tr key={domain}>
                    <td>{domain}</td>
                    <td>{count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  );
}
