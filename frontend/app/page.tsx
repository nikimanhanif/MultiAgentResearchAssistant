import { DashboardLayout } from '@/components/layout/dashboard-layout'
import { ChatContainer } from '@/components/chat/chat-container'

export default function Home() {
  return (
    <DashboardLayout>
      <ChatContainer />
    </DashboardLayout>
  )
}
