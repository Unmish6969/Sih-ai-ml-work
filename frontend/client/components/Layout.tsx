import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';

export default function Layout() {
    return (
        <div className="flex h-screen w-full bg-slate-50">
            <Sidebar />
            {/* <main className = "flex-1 overflow-y-auto p-6"> */}
                <main className="flex-1 overflow-y-auto p-0">

                <Outlet />
            </main>
        </div>
    );
}