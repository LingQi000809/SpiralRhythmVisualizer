import ComparisonPage from './pages/ComparisonPage';

export default function App() {
  return (
    <div style={shell}>
      <ComparisonPage />
    </div>
  );
}

const shell: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  width: '100vw',
  height: '100vh',
  background: '#0b0e14',
  color: '#fff',
  fontFamily: 'Inter, sans-serif',
  boxSizing: 'border-box',
  overflow: 'hidden',
  padding: '20px',
};
