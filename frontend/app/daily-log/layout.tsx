import { Navigation } from '@/components/Navigation';

export default function DailyLogLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen">
      <Navigation />
      <div className="md:ml-20 lg:ml-64 pt-16 md:pt-0 pb-20 md:pb-0">
        {children}
      </div>
    </div>
  );
}