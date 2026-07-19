import { Skeleton } from "@/components/ui/skeleton";

export function AnswerSkeleton() {
  return (
    <div className="space-y-4">
      <div className="glass-card rounded-2xl p-6">
        <Skeleton className="h-5 w-32 rounded-full" />
        <div className="mt-4 space-y-2.5">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-[92%]" />
          <Skeleton className="h-4 w-[70%]" />
        </div>
        <div className="mt-5 flex gap-2">
          <Skeleton className="h-6 w-24 rounded-full" />
          <Skeleton className="h-6 w-28 rounded-full" />
        </div>
      </div>
      <div className="grid gap-3 lg:hidden">
        {[0, 1].map((i) => (
          <div key={i} className="glass-card rounded-xl p-4">
            <div className="flex gap-3">
              <Skeleton className="h-7 w-7 rounded-lg" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-3 w-40" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-12 w-full rounded-lg" />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function CitationSkeletonList() {
  return (
    <div className="space-y-3">
      {[0, 1, 2].map((i) => (
        <div key={i} className="glass-card rounded-xl p-4">
          <div className="flex gap-3">
            <Skeleton className="h-7 w-7 rounded-lg" />
            <div className="flex-1 space-y-2">
              <Skeleton className="h-3 w-40" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-14 w-full rounded-lg" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
