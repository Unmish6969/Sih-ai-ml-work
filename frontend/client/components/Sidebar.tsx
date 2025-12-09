// import { Wind, TrendingUp, BarChart3, FileText } from 'lucide-react';
// import { useNavigate, useLocation } from 'react-router-dom';
// import { useTheme } from '@/contexts/ThemeContext';

// interface NavItem {
//   id: string;
//   label: string;
//   icon: React.ReactNode;
//   path: string;
// }

// interface SidebarProps {
//   activeNav?: string;
//   onNavChange?: (id: string) => void;
// }

// export default function Sidebar({ activeNav = 'dashboard', onNavChange }: SidebarProps) {
//   const { theme } = useTheme();
//   const navigate = useNavigate();
//   const location = useLocation();

//   const navItems: NavItem[] = [
//     { id: 'dashboard', label: 'Dashboard', icon: <Wind className="w-5 h-5" />, path: '/' },
//     { id: 'forecast', label: 'Forecast', icon: <TrendingUp className="w-5 h-5" />, path: '/' },
//     { id: 'evaluation', label: 'Model Evaluation', icon: <BarChart3 className="w-5 h-5" />, path: '/' },
//     { id: 'reports', label: 'Reports', icon: <FileText className="w-5 h-5" />, path: '/reports' },
//   ];

//   const handleNavClick = (item: NavItem) => {
//     onNavChange?.(item.id);
//     navigate(item.path);
//   };

//   const isActive = (path: string, id: string) => {
//     if (id === 'reports') return location.pathname === '/reports';
//     return location.pathname === '/';
//   };

//   return (
//     <aside className={`w-64 flex flex-col h-screen sticky top-0 border-r ${theme === 'dark' ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200'}`}>
//       {/* Logo */}
//       <div className={`p-6 border-b ${theme === 'dark' ? 'border-slate-700' : 'border-slate-200'}`}>
//         <div
//           className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition"
//           onClick={() => navigate('/')}
//         >
//           <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
//             <Wind className="w-6 h-6 text-white" />
//           </div>
//           <span className={`font-bold text-lg ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>SkySense</span>
//         </div>
//       </div>

//       {/* Navigation */}
//       <nav className="flex-1 px-4 py-6 space-y-2">
//         {navItems.map((item) => (
//           <button
//             key={item.id}
//             onClick={() => handleNavClick(item)}
//             className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all text-sm font-medium ${
//               isActive(item.path, item.id)
//                 ? 'bg-cyan-600 text-white'
//                 : theme === 'dark'
//                 ? 'text-slate-300 hover:bg-slate-800'
//                 : 'text-slate-600 hover:bg-slate-100'
//             }`}
//           >
//             {item.icon}
//             {item.label}
//           </button>
//         ))}
//       </nav>

//       {/* Footer */}
//       <div className={`p-4 border-t ${theme === 'dark' ? 'border-slate-700' : 'border-slate-200'}`}>
//         <p className={`text-xs ${theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}`}>Air Quality Monitoring System</p>
//       </div>
//     </aside>
//   );
// }

import { Wind, TrendingUp, BarChart3, FileText, ChevronLeft, ChevronRight, Info ,LayoutDashboard } from 'lucide-react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useTheme } from '@/contexts/ThemeContext';
import { useState } from 'react';

interface NavItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  path: string;
}

interface SidebarProps {
  activeNav?: string;
  onNavChange?: (id: string) => void;
}

export default function Sidebar({ activeNav = 'dashboard', onNavChange }: SidebarProps) {
  const { theme } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const [isCollapsed, setIsCollapsed] = useState(false);

  const navItems: NavItem[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard className="w-5 h-5" />, path: '/' },
    { id: 'forecast', label: 'Forecast', icon: <TrendingUp className="w-5 h-5" />, path: '/forecast' },
    { id: 'evaluation', label: 'Model Evaluation', icon: <BarChart3 className="w-5 h-5" />, path: '/model_evaluation' },
    { id: 'reports', label: 'Reports', icon: <FileText className="w-5 h-5" />, path: '/reports' },
    { id: 'about', label: 'About', icon: <Info className="w-5 h-5" />, path: '/about' },
    
  ];

  const handleNavClick = (item: NavItem) => {
    onNavChange?.(item.id);
    navigate(item.path);

    setTimeout(() => {
    const mainElement = document.querySelector('main');
    if (mainElement) {
      mainElement.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    }
  }, 200);
  };

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <aside 
      className={`flex flex-col h-screen sticky top-0 border-r transition-all duration-300 ${
        isCollapsed ? 'w-20' : 'w-64'
      } ${theme === 'dark' ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200'}`}
    >
      {/* Logo */}
      <div className={`p-6 border-b ${theme === 'dark' ? 'border-slate-700' : 'border-slate-200'} relative`}>
        <div
          className={`flex items-center gap-3 cursor-pointer hover:opacity-80 transition ${
            isCollapsed ? 'justify-center' : ''
          }`}
          onClick={() => navigate('/')}
        >
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center flex-shrink-0">
            <Wind className="w-6 h-6 text-white" />
          </div>
          {!isCollapsed && (
            <span className={`font-bold text-lg whitespace-nowrap ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
              SkySense
            </span>
          )}
        </div>
        
        {/* Toggle Button */}
        <button
          onClick={toggleSidebar}
          className={`absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full border flex items-center justify-center transition-colors ${
            theme === 'dark' 
              ? 'bg-slate-800 border-slate-700 hover:bg-slate-700' 
              : 'bg-white border-slate-300 hover:bg-slate-100'
          }`}
        >
          {isCollapsed ? (
            <ChevronRight className={`w-4 h-4 ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`} />
          ) : (
            <ChevronLeft className={`w-4 h-4 ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`} />
          )}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-2 overflow-hidden">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => handleNavClick(item)}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all text-sm font-medium ${
              isActive(item.path)
                ? 'bg-cyan-600 text-white'
                : theme === 'dark'
                ? 'text-slate-300 hover:bg-slate-800'
                : 'text-slate-600 hover:bg-slate-100'
            } ${isCollapsed ? 'justify-center' : ''}`}
            title={isCollapsed ? item.label : undefined}
          >
            <span className="flex-shrink-0">{item.icon}</span>
            {!isCollapsed && <span className="whitespace-nowrap">{item.label}</span>}
          </button>
        ))}
      </nav>

      {/* Footer */}
      {!isCollapsed && (
        <div className={`p-4 border-t ${theme === 'dark' ? 'border-slate-700' : 'border-slate-200'}`}>
          <p className={`text-xs ${theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}`}>
            Air Quality Monitoring System
          </p>
        </div>
      )}
    </aside>
  );
}
