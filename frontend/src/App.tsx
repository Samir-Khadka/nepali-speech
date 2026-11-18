import { HashRouter, Link, NavLink, Route, Routes } from 'react-router-dom';
import RecordPage from './pages/RecordPage';
import ReviewPage from './pages/ReviewPage';
import StatsPage from './pages/StatsPage';

export default function App() {
  return (
    <HashRouter>
      <header className="app-header">
        <div className="app-header-inner">
          <p className="app-title">
            🎙️ <span>नेपाली</span> Speech Collector
          </p>
          <nav className="app-nav" aria-label="Main navigation">
            <NavLink to="/" end>
              रेकर्ड गर्नुहोस्
            </NavLink>
            <NavLink to="/review">समीक्षा</NavLink>
            <NavLink to="/stats">तथ्याङ्क</NavLink>
          </nav>
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<RecordPage />} />
          <Route path="/review" element={<ReviewPage />} />
          <Route path="/stats" element={<StatsPage />} />
          <Route
            path="*"
            element={
              <div className="card">
                <h1>पृष्ठ भेटिएन</h1>
                <p>
                  Page not found. <Link to="/">Go back to recording</Link>.
                </p>
              </div>
            }
          />
        </Routes>
      </main>
    </HashRouter>
  );
}
