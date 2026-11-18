import { useState } from 'react';
import { ApiError, fetchAdminStats, type AdminStats } from '../api';

/** AdminStats plus the Phase 2 coverage fields from GET /api/admin/stats. */
interface CoverageStats extends AdminStats {
  by_province: Record<string, number>;
  by_gender: Record<string, number>;
  by_age_range: Record<string, number>;
  hours_estimate: {
    approved_hours: number;
    recorded_hours: number;
    hours_basis: string;
  };
}

function sortedEntries(record: Record<string, number>) {
  return Object.entries(record).sort((a, b) => b[1] - a[1]);
}

export default function StatsPage() {
  const [email, setEmail] = useState<string>(
    () => localStorage.getItem('ns-admin-email') ?? '',
  );
  const [stats, setStats] = useState<CoverageStats | null>(null);
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
      const data = (await fetchAdminStats(adminEmail)) as CoverageStats;
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

          <div className="card">
            <h2>अनुमानित घण्टा · Hours (estimate)</h2>
            <p className="muted">{stats.hours_estimate.hours_basis}</p>
            <div className="stat-grid">
              <div className="stat-chip">
                <strong>{stats.hours_estimate.approved_hours}</strong>
                स्वीकृत घण्टा (अनुमान) · Approved hours (estimate)
              </div>
              <div className="stat-chip">
                <strong>{stats.hours_estimate.recorded_hours}</strong>
                रेकर्ड गरिएका घण्टा (अनुमान) · Recorded hours (estimate)
              </div>
            </div>
          </div>

          <div className="card">
            <h2>प्रदेश अनुसार · By province</h2>
            <table className="stats-table">
              <tbody>
                {sortedEntries(stats.by_province).map(([province, count]) => (
                  <tr key={province}>
                    <td>{province}</td>
                    <td>{count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card">
            <h2>लिङ्ग अनुसार · By gender</h2>
            <table className="stats-table">
              <tbody>
                {sortedEntries(stats.by_gender).map(([gender, count]) => (
                  <tr key={gender}>
                    <td>{gender}</td>
                    <td>{count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card">
            <h2>उमेर समूह अनुसार · By age range</h2>
            <table className="stats-table">
              <tbody>
                {sortedEntries(stats.by_age_range).map(([ageRange, count]) => (
                  <tr key={ageRange}>
                    <td>{ageRange}</td>
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
