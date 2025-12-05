import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'MigraineMamba | AI-Powered Migraine Prediction',
  description: 'Predict migraine attacks 24 hours in advance using advanced AI and personalized trigger analysis.',
  keywords: ['migraine', 'prediction', 'AI', 'health', 'headache', 'triggers'],
  authors: [{ name: 'MigraineMamba Team' }],
  openGraph: {
    title: 'MigraineMamba | AI-Powered Migraine Prediction',
    description: 'Predict migraine attacks 24 hours in advance using advanced AI.',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="font-sans">
        <div className="relative min-h-screen">
          {/* Background effects */}
          <div className="fixed inset-0 grid-pattern pointer-events-none" />
          <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-neural-500/10 rounded-full blur-[120px] pointer-events-none" />
          <div className="fixed bottom-0 right-0 w-[400px] h-[400px] bg-purple-500/10 rounded-full blur-[100px] pointer-events-none" />
          
          {/* Main content */}
          <main className="relative z-10">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}